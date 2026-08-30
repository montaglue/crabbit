#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "harness.py --backend.*20260831-clean-full" >/dev/null; do sleep 120; done
echo "harness exited at $(date)" >> postprocess.log
python3 /home/reg/projects/montaglue/kernel-corpus/tools/cpu_analysis.py results.csv --out . >> postprocess.log 2>&1
echo "analysis written at $(date)" >> postprocess.log
