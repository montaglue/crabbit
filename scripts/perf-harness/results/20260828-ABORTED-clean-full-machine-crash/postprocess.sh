#!/bin/bash
# Detached watcher: when the clean sweep's harness exits, render the CPU
# analysis into this directory. Started 2026-08-28 15:35 with nohup.
cd "$(dirname "$0")"
while pgrep -f "harness.py --skip-backend-build --runs 20 --pin-cpu 7" >/dev/null; do sleep 120; done
echo "harness exited at $(date)" >> postprocess.log
python3 /home/reg/projects/montaglue/kernel-corpus/tools/cpu_analysis.py results.csv --out . >> postprocess.log 2>&1
echo "analysis written at $(date)" >> postprocess.log
