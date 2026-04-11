#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p results/pilot
nohup python3 -u code/run_pilot.py "$@" > results/pilot/pilot_run.log 2>&1 &
echo "PID: $!"
echo "Log: tail -f results/pilot/pilot_run.log"
