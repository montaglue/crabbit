#!/bin/bash
# Run a long/overnight job under a hard memory cap so it can never take the
# machine down: the kernel OOM-kills ONLY this job's cgroup at the cap, and
# choom makes this job the kernel's first victim even outside the cap.
# No Docker — plain cgroup v2 via a user systemd scope.
#
# usage: scripts/run-capped.sh [MEMMAX] [NAME] -- command args...
#   MEMMAX default 80G (of 121G; leaves the system >40G headroom)
# example:
#   scripts/run-capped.sh 60G sweep -- python3 scripts/perf-harness/harness.py ...
set -euo pipefail
MEM="${1:-80G}"; NAME="${2:-capped-job}"
shift 2 || true
[ "${1:-}" = "--" ] && shift
exec systemd-run --user --scope --unit="$NAME-$$" \
  -p MemoryMax="$MEM" -p MemorySwapMax=4G -p MemoryHigh=$(numfmt --from=iec "$MEM" | awk '{printf "%d", $1*0.9}') \
  -p TasksMax=4096 \
  choom -n 500 -- "$@"
