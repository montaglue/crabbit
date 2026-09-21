//! Spectral block execution frequencies.
//!
//! Ported from combinatorial-matrix-theory Experiment A: with uniform
//! branch priors the CFG is a row-substochastic transition matrix; closing
//! the chain with exit→entry restart edges makes it irreducible, so
//! Perron–Frobenius gives a unique stationary distribution π, found by
//! power iteration on the lazy chain `(P + I) / 2` (lazy for
//! aperiodicity). The compiler-meaningful quantity is the
//! renewal-normalized `freq(b) = π(b) / π(entry)`: expected executions of
//! `b` per function invocation. Entry gets 1, the two sides of a diamond
//! get 0.5 each, a loop body with backedge probability p gets `1/(1−p)`.
//!
//! This is purely analytical — no profile, no measurement — which is what
//! lets it ride in the backend as a cost-model option (see
//! [crate::codegen_opts]). It is deliberately graph-shaped (successor
//! lists in, frequencies out) so any backend can feed it its machine CFG.

/// Default expected trip count `N` of the loop-aware prior
/// ([spectral_frequencies_loop_aware]): a backedge carries probability
/// `q = 1 - 1/N` (LLVM's loop-branch-weight convention), so a loop header
/// runs `1/(1-q) = N` times per entry into the loop.
const DEFAULT_LOOP_TRIP_COUNT: f64 = 8.0;

/// Expected executions of each block per function invocation, under the
/// uniform branch prior.
///
/// `successors[i]` lists the successor indices of block `i`; `entry` is the
/// function entry block. Blocks unreachable from `entry` get frequency 0.
pub fn spectral_frequencies(successors: &[Vec<usize>], entry: usize) -> Vec<f64> {
    let weights: Vec<Vec<(usize, f64)>> = successors
        .iter()
        .map(|succs| {
            let share = 1.0 / succs.len() as f64;
            succs.iter().map(|&succ| (succ, share)).collect()
        })
        .collect();
    stationary_frequencies(&weights, entry)
}

/// [spectral_frequencies] with a loop-aware branch prior: an edge `b → h`
/// where `h` dominates `b` is a backedge (dominance is reflexive, so
/// self-loops count), backedges define natural loops, and a branch inside
/// a loop gives the successors that *stay in the branch block's innermost
/// loop* probability `q = 1 - 1/N` (`N = 8`, so a self-loop header gets
/// frequency 8 and each nesting level multiplies by 8) with the
/// loop-leaving successors sharing the remainder; every other branch
/// keeps the uniform prior. This is LLVM's loop-branch-weight convention
/// (`BranchProbabilityInfo`'s loop heuristic): a branch whose taken edge
/// is itself a backedge is the special case where the taken edge targets
/// the header, but this backend's machine CFGs rotate loops so the
/// backedge sits on an unconditional latch (probability 1 regardless of
/// prior) and the iteration decision is the header's in-loop vs exit
/// branch — which this prior weights and a literal backedge-edge rule
/// measurably misses (it was a no-op on the whole kernel-corpus E4
/// profile set). A CFG whose cycles have no dominance-detected backedge
/// (irreducible control flow) degrades gracefully to exactly the
/// uniform-prior frequencies.
pub fn spectral_frequencies_loop_aware(successors: &[Vec<usize>], entry: usize) -> Vec<f64> {
    let n = successors.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(entry < n, "entry block index out of range");
    let dom = dominator_sets(successors, entry);

    // Natural loop bodies: for each backedge source → header, the body is
    // the header plus every block that reaches the source against the
    // edges without passing through the header. Backedges sharing a
    // header merge into one body. Unreachable blocks have empty dominator
    // sets, so they spawn no loops — harmless, since the chain never puts
    // mass on them.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (block, succs) in successors.iter().enumerate() {
        for &succ in succs {
            if !preds[succ].contains(&block) {
                preds[succ].push(block);
            }
        }
    }
    let mut bodies: Vec<(usize, Vec<bool>)> = Vec::new(); // (header, body)
    for (source, succs) in successors.iter().enumerate() {
        for &header in succs {
            if !dom[source].contains(&header) {
                continue; // not a backedge
            }
            let slot = match bodies.iter().position(|(h, _)| *h == header) {
                Some(slot) => slot,
                None => {
                    let mut body = vec![false; n];
                    body[header] = true;
                    bodies.push((header, body));
                    bodies.len() - 1
                }
            };
            let body = &mut bodies[slot].1;
            let mut stack = vec![source];
            while let Some(x) = stack.pop() {
                if body[x] {
                    continue;
                }
                body[x] = true;
                stack.extend(preds[x].iter().copied());
            }
        }
    }
    // Innermost first: the smallest body containing a block is its loop.
    bodies.sort_by_key(|(_, body)| body.iter().filter(|&&x| x).count());

    let q = 1.0 - 1.0 / DEFAULT_LOOP_TRIP_COUNT;
    let weights: Vec<Vec<(usize, f64)>> = successors
        .iter()
        .enumerate()
        .map(|(block, succs)| {
            let k = succs.len();
            let innermost = bodies.iter().find(|(_, body)| body[block]);
            let stays: Vec<bool> = match innermost {
                Some((_, body)) => succs.iter().map(|&succ| body[succ]).collect(),
                None => vec![false; k],
            };
            let n_stay = stays.iter().filter(|&&s| s).count();
            if n_stay == 0 || n_stay == k {
                let share = 1.0 / k as f64;
                succs.iter().map(|&succ| (succ, share)).collect()
            } else {
                let w_stay = q / n_stay as f64;
                let w_leave = (1.0 - q) / (k - n_stay) as f64;
                succs
                    .iter()
                    .zip(&stays)
                    .map(|(&succ, &stay)| (succ, if stay { w_stay } else { w_leave }))
                    .collect()
            }
        })
        .collect();
    stationary_frequencies(&weights, entry)
}

/// Shared power-iteration core: stationary distribution of the weighted
/// chain closed with exit→entry restart edges, renewal-normalized to
/// `freq(b) = π(b)/π(entry)`.
///
/// `weights[i]` lists `(successor, probability)` pairs summing to 1 for
/// non-exit blocks; an empty row is an exit and restarts at entry. The
/// iteration is on the lazy chain `(P + I)/2` so periodic CFGs (e.g. a
/// two-block cycle) still converge.
fn stationary_frequencies(weights: &[Vec<(usize, f64)>], entry: usize) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(entry < n, "entry block index out of range");

    // π starts with all mass at entry; one step of the closed chain moves
    // mass along the weighted edges, exits (and any dead-end) restart at
    // entry.
    let mut pi = vec![0.0f64; n];
    pi[entry] = 1.0;
    let mut next = vec![0.0f64; n];
    for _ in 0..10_000 {
        next.fill(0.0);
        for (block, edges) in weights.iter().enumerate() {
            let mass = pi[block];
            if mass == 0.0 {
                continue;
            }
            if edges.is_empty() {
                next[entry] += mass;
            } else {
                for &(succ, w) in edges {
                    next[succ] += mass * w;
                }
            }
        }
        let mut delta = 0.0;
        for i in 0..n {
            let lazy = 0.5 * pi[i] + 0.5 * next[i];
            delta += (lazy - pi[i]).abs();
            next[i] = lazy;
        }
        std::mem::swap(&mut pi, &mut next);
        if delta < 1e-12 {
            break;
        }
    }

    let entry_mass = pi[entry];
    if entry_mass <= 0.0 {
        // Degenerate chain (should not happen with the restart closure):
        // fall back to uniform weights so consumers stay well-defined.
        return vec![1.0; n];
    }
    pi.iter().map(|&mass| mass / entry_mass).collect()
}

/// Dominator sets over a plain index graph: `dom[b]` is the set of blocks
/// dominating `b` (reflexive for reachable `b`; empty for unreachable
/// `b`). The classic iterative fixpoint
/// `D(entry) = {entry}; D(b) = {b} ∪ ⋂ D(preds)` as a bitset sweep — these
/// CFGs are tiny, so plain repeated sweeps to fixpoint are plenty.
fn dominator_sets(successors: &[Vec<usize>], entry: usize) -> Vec<Vec<usize>> {
    let n = successors.len();
    let words = n.div_ceil(64);

    // Reachability from entry (DFS), and reachable predecessor lists.
    let mut reachable = vec![false; n];
    let mut stack = vec![entry];
    reachable[entry] = true;
    while let Some(block) = stack.pop() {
        for &succ in &successors[block] {
            if !reachable[succ] {
                reachable[succ] = true;
                stack.push(succ);
            }
        }
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (block, succs) in successors.iter().enumerate() {
        if !reachable[block] {
            continue;
        }
        for &succ in succs {
            if !preds[succ].contains(&block) {
                preds[succ].push(block);
            }
        }
    }

    // Init: entry = {entry}; other reachable rows = ⊤; unreachable = ∅.
    // Rows only ever shrink, so the iteration reaches the greatest
    // fixpoint — the true dominator sets.
    let full = vec![u64::MAX; words];
    let mut dom: Vec<Vec<u64>> = (0..n)
        .map(|b| {
            if b == entry {
                let mut row = vec![0u64; words];
                row[entry / 64] |= 1 << (entry % 64);
                row
            } else if reachable[b] {
                full.clone()
            } else {
                vec![0u64; words]
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in 0..n {
            if b == entry || !reachable[b] {
                continue;
            }
            // acc = {b} ∪ ⋂_{p ∈ preds(b)} D(p); every reachable non-entry
            // block has at least one reachable predecessor.
            let mut acc = full.clone();
            for &p in &preds[b] {
                for (a, d) in acc.iter_mut().zip(&dom[p]) {
                    *a &= d;
                }
            }
            acc[b / 64] |= 1 << (b % 64);
            if acc != dom[b] {
                dom[b] = acc;
                changed = true;
            }
        }
    }

    dom.iter()
        .map(|row| {
            (0..n)
                .filter(|&j| (row[j / 64] >> (j % 64)) & 1 == 1)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{spectral_frequencies, spectral_frequencies_loop_aware};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn straight_line_is_all_ones() {
        let freqs = spectral_frequencies(&[vec![1], vec![2], vec![]], 0);
        for freq in freqs {
            assert_close(freq, 1.0);
        }
    }

    #[test]
    fn diamond_sides_get_half() {
        // 0 → {1, 2} → 3
        let freqs = spectral_frequencies(&[vec![1, 2], vec![3], vec![3], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.5);
        assert_close(freqs[2], 0.5);
        assert_close(freqs[3], 1.0);
    }

    #[test]
    fn loop_body_gets_expected_trip_count() {
        // 0 → 1; 1 → {1, 2}: backedge probability 0.5 → E[visits] = 2.
        let freqs = spectral_frequencies(&[vec![1], vec![1, 2], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 2.0);
        assert_close(freqs[2], 1.0);
    }

    #[test]
    fn unreachable_block_gets_zero() {
        let freqs = spectral_frequencies(&[vec![], vec![0]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.0);
    }

    /// Loop-aware prior, self-loop: the backedge 1 → 1 carries
    /// q = 1 - 1/8 = 7/8, so header visits are geometric with escape
    /// probability 1/8: freq = 1/(1-q) = 8 exactly (entry feeds the loop
    /// once per invocation, so the renewal normalization adds no factor).
    #[test]
    fn loop_aware_self_loop_gets_trip_count() {
        let freqs = spectral_frequencies_loop_aware(&[vec![1], vec![1, 2], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 8.0);
        assert_close(freqs[2], 1.0);
    }

    /// Loop-aware prior, nested loop: 0 → 1(outer) → 2(inner);
    /// 2 → {2, 3}; 3(latch) → {1, 4}; 4 = exit. Backedges 2→2 and 3→1
    /// carry 7/8 each. Visit equations per invocation:
    /// f1 = 1 + 7/8·f3; f2 = f1 + 7/8·f2 ⇒ f2 = 8·f1; f3 = 1/8·f2 = f1
    /// ⇒ f1 = 8, f2 = 64, f3 = 8, f4 = 1/8·f3 = 1.
    #[test]
    fn loop_aware_nested_loop_compounds() {
        let freqs = spectral_frequencies_loop_aware(
            &[vec![1], vec![2], vec![2, 3], vec![1, 4], vec![]],
            0,
        );
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 8.0);
        assert_close(freqs[2], 64.0);
        assert_close(freqs[3], 8.0);
        assert_close(freqs[4], 1.0);
    }

    /// Rotated (latch-form) loop — the shape this backend's machine CFGs
    /// actually produce: 0 → 1(header); 1 → {2(body), 3(exit)};
    /// 2 → 1 (the backedge, on the unconditional latch). The natural loop
    /// is {1, 2} and the header's in-loop edge carries 7/8:
    /// f1 = 1 + f2, f2 = 7/8·f1 ⇒ f1 = 8, f2 = 7, f3 = 1/8·f1 = 1.
    /// A literal "taken edge is a backedge" rule would leave the header's
    /// branch uniform (the backedge is the latch's only edge) and change
    /// nothing.
    #[test]
    fn loop_aware_rotated_loop_weights_the_header_branch() {
        let freqs =
            spectral_frequencies_loop_aware(&[vec![1], vec![2, 3], vec![1], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 8.0);
        assert_close(freqs[2], 7.0);
        assert_close(freqs[3], 1.0);
    }

    /// Irreducible cycle {1, 2} with two entries: neither dominates the
    /// other, no backedge is found, and the loop-aware frequencies
    /// degrade gracefully to exactly the uniform-prior ones.
    #[test]
    fn loop_aware_irreducible_falls_back_to_uniform() {
        let cfg: &[Vec<usize>] = &[vec![1, 2], vec![2, 3], vec![1, 3], vec![]];
        let uniform = spectral_frequencies(cfg, 0);
        let aware = spectral_frequencies_loop_aware(cfg, 0);
        for (u, a) in uniform.iter().zip(&aware) {
            assert_close(*a, *u);
        }
    }

    /// Loop-aware prior leaves unreachable blocks at zero and the diamond
    /// (no cycles at all) at the uniform answer.
    #[test]
    fn loop_aware_acyclic_and_unreachable_match_uniform() {
        let freqs = spectral_frequencies_loop_aware(&[vec![1, 2], vec![3], vec![3], vec![]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.5);
        assert_close(freqs[2], 0.5);
        assert_close(freqs[3], 1.0);
        let freqs = spectral_frequencies_loop_aware(&[vec![], vec![0]], 0);
        assert_close(freqs[0], 1.0);
        assert_close(freqs[1], 0.0);
    }
}
