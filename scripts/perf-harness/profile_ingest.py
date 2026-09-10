#!/usr/bin/env python3
"""perf samples + crabbit blockmaps -> profile.json (docs/PROFILE-FEEDBACK-PLAN.md).

Turns `perf record` samples into the measured block-frequency profile the
crabbit backend consumes under `CRABBIT_BLOCK_FREQ=profile` +
`CRABBIT_PROFILE=<profile.json>`.

Pipeline:

  1. Build the target with `CRABBIT_BLOCKMAP=1` -> the backend writes a
     `<object stem>.blockmap.json` next to each object:
     `{symbol: [{id, start, end}, ...]}` where `id` is the block's index in
     RA-time region order (-1 for blocks created after RA) and
     `start`/`end` are final `.text` byte offsets in layout order.
  2. `perf record -e cycles:u [-c PERIOD] -- <bin> ...`
  3. `perf script -F ip,sym,symoff [--no-demangle] > samples.txt`
     (`--no-demangle` keeps the symbol spelling identical to the
     blockmap's; plain Rust `#[no_mangle]` symbols match either way)
  4. `profile_ingest.py --blockmap a.blockmap.json [--blockmap ...]
         --perf-script samples.txt [--perf-script more-runs.txt ...]
         -o profile.json`

Sample attribution: a `sym+0xOFF` line maps to the absolute `.text` offset
`base(sym) + OFF` (the symbol's base is the minimum `start` of its ranges,
since a function's ranges cover it contiguously), then to the range
containing that offset, then to that range's RA-order block id. Samples in
unknown symbols, outside any range, or in `id == -1` ranges are dropped
(and counted in the stderr summary). Symbols present in the profile but
missing from a stale blockmap - or vice versa - are never an error: the
backend falls back to uniform per function.

Normalization mirrors the backend's spectral model
(`crates/pliron-ll/src/passes/spectral_freq.rs`): per function,
`freq[b] = samples[b] / samples[entry]` with the entry (id 0) forced to at
least one sample, unsampled blocks 0.0, so entry == 1.0 always. Multiple
`--perf-script` runs merge additively before normalizing.

Fallback for perf builds whose `script` lacks symbol output: pass
`--nm-binary <bin>` (and optionally `--load-bias 0x...`) to resolve bare
`ip`-only lines against `nm`'s symbol table. This only matches when the
recorded addresses equal the link-time addresses plus the given bias
(non-PIE/static binaries, or a bias read from `perf script -F ip,dso` /
`/proc/<pid>/maps`).

Op-level backward attribution (docs/PROFILE-FEEDBACK-BACKWARD.md): when
the blockmap was written with op stamping (ranges carry an `ops` array of
`{index, derived_from, start, end}`), the tool also emits
`<output stem>.op_costs.json` (or `--op-costs PATH`):

  { "<symbol>": {
      "machine": {"<op index>": samples, ...},
      "lifted":  {"<llvm op id | root name>": samples, ...} } }

`machine` is raw per-instruction sample counts (final layout order);
`source` (present when the blockmap carries a `__midend__` table) is the
second hop: `lifted` ids that the mid-end created are split (equal
weights) onto their source parents, so keys are PRE-mid-end op ids or
root names, with `midend` for mid-end ops that declared no adjoint.
`lifted` is the first backward hop - the same samples aggregated onto the
LLVM-level op ids each instruction was lowered from, with synthetic
per-pass roots (`isel:abi`, `regalloc`, `frame`, `placement`,
`unattributed`) for code created from nothing.

Stdlib only.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import subprocess
import sys
from collections import defaultdict


# `perf script -F ip,sym,symoff`: "<ip-hex> <symbol>+0x<off>". The symbol
# may contain spaces (demangled C++/Rust), so anchor the offset at the end.
SYM_OFF_RE = re.compile(r"^\s*(?:([0-9a-fA-F]+)\s+)?(.+?)\+0x([0-9a-fA-F]+)\s*$")
IP_ONLY_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s*$")


# Synthetic derived_from roots (crates/pliron-ll/src/passes/aarch64/opmap.rs).
# Keep in sync with `pub mod roots` in
# crates/pliron-ll/src/passes/aarch64/opmap.rs (test_root_names_cover_rust_roots).
ROOT_NAMES = {
    -1: "isel:abi",
    -2: "regalloc",
    -3: "frame",
    -4: "placement",
    -5: "unattributed",
    -6: "midend",
}


def lifted_key(derived_from):
    if derived_from >= 0:
        return str(derived_from)
    return ROOT_NAMES.get(derived_from, f"root:{derived_from}")


def load_midend_tables(paths):
    """{symbol: {fresh RA-boundary id: [source parent ids]}} merged from
    the blockmaps' `__midend__` entries."""
    merged = {}
    for path in paths:
        with open(path) as handle:
            raw = json.load(handle)
        for symbol, table in raw.get("__midend__", {}).items():
            merged.setdefault(symbol, {}).update(
                {int(k): [int(x) for x in v] for k, v in table.items()}
            )
    return merged


def lift_to_source(lifted, midend_table):
    """Second backward hop: weighted (equal) split of mid-end-created ids
    onto their source parents. Ids not in the table are already source
    ids; roots pass through. Returns {id-or-root: float samples}."""
    out = defaultdict(float)
    for source, count in lifted.items():
        parents = midend_table.get(source) if source >= 0 else None
        if not parents:
            out[source] += count
            continue
        share = count / len(parents)
        for parent in parents:
            out[parent] += share
    return out


def load_blockmaps(paths):
    """Merge blockmap files into {symbol: {"ranges": [(start, end, id)...],
    "base": min start, "blocks": max id + 1}}. Ranges are sorted by start.
    A symbol defined by several objects keeps its first definition (with a
    warning): duplicate symbols cannot be told apart in a perf trace."""
    merged = {}
    for path in paths:
        with open(path) as handle:
            raw = json.load(handle)
        for symbol, ranges in raw.items():
            if symbol == "__midend__":
                continue
            if symbol in merged:
                print(
                    f"profile_ingest: warning: symbol `{symbol}` appears in more than "
                    f"one blockmap; keeping the first definition",
                    file=sys.stderr,
                )
                continue
            parsed = sorted(
                (int(r["start"]), int(r["end"]), int(r["id"])) for r in ranges
            )
            if not parsed:
                continue
            ids = [r[2] for r in parsed if r[2] >= 0]
            if not ids:
                continue
            op_ranges = sorted(
                (
                    int(op["start"]),
                    int(op["end"]),
                    int(op["index"]),
                    int(op["derived_from"]),
                )
                for r in ranges
                for op in r.get("ops", [])
            )
            merged[symbol] = {
                "ranges": parsed,
                "base": parsed[0][0],
                "blocks": max(ids) + 1,
                "op_ranges": op_ranges,
            }
    return merged


def load_nm_table(binary, load_bias):
    """[(address, symbol)] sorted by address, from `nm --defined-only`."""
    out = subprocess.run(
        ["nm", "--defined-only", binary],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    table = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].lower() in ("t", "w"):
            table.append((int(parts[0], 16) + load_bias, parts[2]))
    table.sort()
    return table


def resolve_ip(nm_table, ip):
    """(symbol, offset) of the nm symbol covering `ip`, or None."""
    if not nm_table:
        return None
    index = bisect.bisect_right(nm_table, (ip, "￿")) - 1
    if index < 0:
        return None
    address, symbol = nm_table[index]
    return symbol, ip - address


def match_symbol(blockmaps, symbol):
    """The blockmap entry for a perf symbol; tolerates the Darwin-style
    leading underscore in either direction."""
    if symbol in blockmaps:
        return symbol
    if symbol.startswith("_") and symbol[1:] in blockmaps:
        return symbol[1:]
    if "_" + symbol in blockmaps:
        return "_" + symbol
    return None


def attribute(blockmaps, symbol, offset, counts, stats, op_counts):
    """Add one sample at `offset` into `symbol` to `counts` (block level)
    and `op_counts` (instruction level, when the blockmap carries ops)."""
    key = match_symbol(blockmaps, symbol)
    if key is None:
        stats["unknown_symbol"] += 1
        return
    entry = blockmaps[key]
    absolute = entry["base"] + offset
    # Instruction-level attribution is independent of block ids (a sample
    # in a post-RA block still has an attributed instruction).
    op_ranges = entry.get("op_ranges") or []
    if op_ranges:
        op_index = (
            bisect.bisect_right(op_ranges, (absolute, float("inf"), 0, 0)) - 1
        )
        if 0 <= op_index and op_ranges[op_index][0] <= absolute < op_ranges[op_index][1]:
            _, _, machine_index, derived_from = op_ranges[op_index]
            op_counts[key]["machine"][machine_index] += 1
            op_counts[key]["lifted"][derived_from] += 1
            stats["op_attributed"] += 1
    ranges = entry["ranges"]
    index = bisect.bisect_right(ranges, (absolute, float("inf"), 0)) - 1
    if index < 0 or not (ranges[index][0] <= absolute < ranges[index][1]):
        stats["out_of_range"] += 1
        return
    block_id = ranges[index][2]
    if block_id < 0:
        stats["post_ra_block"] += 1
        return
    counts[key][block_id] += 1
    stats["attributed"] += 1


def ingest_lines(lines, blockmaps, counts, stats, op_counts=None, nm_table=None):
    if op_counts is None:
        op_counts = defaultdict(
            lambda: {"machine": defaultdict(int), "lifted": defaultdict(int)}
        )
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = SYM_OFF_RE.match(line)
        if match:
            symbol = match.group(2).strip()
            if symbol == "[unknown]":
                stats["unknown_symbol"] += 1
                continue
            attribute(
                blockmaps, symbol, int(match.group(3), 16), counts, stats, op_counts
            )
            continue
        ip_match = IP_ONLY_RE.match(line)
        if ip_match and nm_table is not None:
            resolved = resolve_ip(nm_table, int(ip_match.group(1), 16))
            if resolved is None:
                stats["unknown_symbol"] += 1
            else:
                attribute(
                    blockmaps, resolved[0], resolved[1], counts, stats, op_counts
                )
            continue
        stats["unparsed"] += 1


def render_op_costs(op_counts, midend_tables=None):
    """JSON-ready op_costs: string keys, lifted roots by name; `source`
    added per symbol when a mid-end table exists for it."""
    out = {}
    for symbol, kinds in sorted(op_counts.items()):
        if not kinds["machine"]:
            continue
        entry = {
            "machine": {
                str(index): count
                for index, count in sorted(kinds["machine"].items())
            },
            "lifted": {
                lifted_key(source): count
                for source, count in sorted(kinds["lifted"].items())
            },
        }
        table = (midend_tables or {}).get(symbol)
        if table:
            source_costs = lift_to_source(kinds["lifted"], table)
            entry["source"] = {
                lifted_key(source): round(count, 3)
                for source, count in sorted(source_costs.items())
            }
        out[symbol] = entry
    return out


def normalize(blockmaps, counts, smooth=1.0):
    """{symbol: [freq per RA-order block index]} for sampled symbols.

    Additive (Laplace) smoothing, default alpha=1: freq[b] =
    (samples[b] + alpha) / (samples[entry] + alpha). A sampled count of
    zero means "below the sampler's resolution", NOT "never executes" —
    a block can run tens of thousands of times too cheaply to catch a
    2 kHz sample. Feeding literal 0.0 into a frequency-weighted spill
    score marks every value used in such blocks as free to evict and
    invites unlimited spill traffic there (measured on gemm_tiled:
    kfn_ldr_sp 115→136 and −3.5% runtime before smoothing). The floor is
    one sample: the detection threshold. `--smooth 0` restores the old
    behavior for comparison."""
    profile = {}
    for symbol, block_counts in counts.items():
        if not block_counts:
            continue
        blocks = blockmaps[symbol]["blocks"]
        entry_samples = max(block_counts.get(0, 0), 1) + smooth
        profile[symbol] = [
            (block_counts.get(block, 0) + smooth) / entry_samples
            for block in range(blocks)
        ]
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="perf script output + crabbit blockmaps -> profile.json"
    )
    parser.add_argument(
        "--blockmap",
        action="append",
        required=True,
        help="a <object stem>.blockmap.json written under CRABBIT_BLOCKMAP=1 "
        "(repeat for multiple objects)",
    )
    parser.add_argument(
        "--perf-script",
        action="append",
        required=True,
        help="output of `perf script -F ip,sym,symoff` ('-' for stdin; "
        "repeat to merge runs additively)",
    )
    parser.add_argument("-o", "--output", required=True, help="profile.json to write")
    parser.add_argument(
        "--op-costs",
        help="where to write the op-level cost JSON (default: "
        "<output stem>.op_costs.json next to --output, only when the "
        "blockmap carries op attribution and samples hit it)",
    )
    parser.add_argument(
        "--nm-binary",
        help="resolve bare-ip lines against this binary's nm symbol table "
        "(fallback when perf script cannot print sym+symoff)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1.0,
        help="additive (Laplace) smoothing per block, in samples; the "
        "default 1.0 floors unsampled blocks at the detection threshold "
        "(a sampled zero means below resolution, not 'never executes'); "
        "0 restores raw counts for A/B comparison",
    )
    parser.add_argument(
        "--load-bias",
        default="0",
        help="runtime load address minus link address, for --nm-binary "
        "(hex ok; default 0)",
    )
    args = parser.parse_args(argv)

    blockmaps = load_blockmaps(args.blockmap)
    nm_table = (
        load_nm_table(args.nm_binary, int(args.load_bias, 0))
        if args.nm_binary
        else None
    )
    counts = defaultdict(lambda: defaultdict(int))
    op_counts = defaultdict(
        lambda: {"machine": defaultdict(int), "lifted": defaultdict(int)}
    )
    stats = defaultdict(int)
    for path in args.perf_script:
        if path == "-":
            ingest_lines(sys.stdin, blockmaps, counts, stats, op_counts, nm_table)
        else:
            with open(path) as handle:
                ingest_lines(handle, blockmaps, counts, stats, op_counts, nm_table)

    profile = normalize(blockmaps, counts, smooth=args.smooth)
    with open(args.output, "w") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")

    midend_tables = load_midend_tables(args.blockmap)
    op_costs = render_op_costs(op_counts, midend_tables)
    if op_costs:
        op_costs_path = args.op_costs or re.sub(
            r"(\.json)?$", ".op_costs.json", args.output, count=1
        )
        with open(op_costs_path, "w") as handle:
            json.dump(op_costs, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(
        "profile_ingest: {attributed} samples attributed across {functions} functions "
        "({op_attr} instruction-attributed); "
        "dropped: {unknown} unknown-symbol, {oor} out-of-range, {post} post-RA-block, "
        "{unparsed} unparsed lines".format(
            attributed=stats["attributed"],
            op_attr=stats["op_attributed"],
            functions=len(profile),
            unknown=stats["unknown_symbol"],
            oor=stats["out_of_range"],
            post=stats["post_ra_block"],
            unparsed=stats["unparsed"],
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
