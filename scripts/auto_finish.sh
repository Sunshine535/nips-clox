#!/bin/bash
# Auto-finish: monitor experiment, push to github, create zip when done
# Usage: nohup bash scripts/auto_finish.sh &

LOG="/home/claude/nips-clox/results/v4/experiment.log"
REPO="/home/claude/nips-clox"
GH="/home/claude/.local/bin/gh"

echo "[$(date)] Monitoring experiment..."

while true; do
    # Check if process is alive
    if ! ps -p 13177 > /dev/null 2>&1; then
        echo "[$(date)] Experiment process ended."
        break
    fi

    # Check if MATH benchmark is complete (27 combos = all strategies done)
    MATH_DONE=$(grep "INFO   " "$LOG" 2>/dev/null | grep "acc=" | grep "tokens=" | wc -l)
    echo "[$(date)] Completed combos: $MATH_DONE"

    # Check if we moved past MATH to strategyqa
    if grep -q "Benchmark: strategyqa" "$LOG" 2>/dev/null && grep -q "Phase 2" "$LOG" 2>/dev/null; then
        STRAT_STARTED=$(grep "Benchmark: strategyqa" "$LOG" | tail -1)
        if echo "$STRAT_STARTED" | grep -q "Phase 2"; then
            echo "[$(date)] StrategyQA strategy phase started - MATH is complete!"
            break
        fi
    fi

    sleep 300  # check every 5 minutes
done

echo "[$(date)] Packaging results..."

cd "$REPO"

# Add all new results
git add -A
git commit -m "$(cat <<'COMMIT'
Add Qwen3-8B experiment results (MATH complete)

- MATH: 300 examples × 9 strategies × 3 seeds (Qwen3-8B, TP=2)
- Topology: 4 benchmarks × 200 examples
- Pilot: 4 benchmarks × 50 examples × 5 strategies
- Synthetic DAG: unchanged

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
COMMIT
)"

# Push
git remote set-url origin https://Sunshine535:$GH_TOKEN@github.com/Sunshine535/nips-clox.git
git push origin main 2>&1
git remote set-url origin https://github.com/Sunshine535/nips-clox.git

# Create zip with everything including results
echo "[$(date)] Creating zip..."
cd /home/claude
zip -r /workspace/nips-clox-results.zip nips-clox/ \
    -x "nips-clox/venv/*" \
    -x "nips-clox/.git/*" \
    -x "nips-clox/__pycache__/*" \
    -x "nips-clox/*/__pycache__/*" \
    2>&1 | tail -3

echo "[$(date)] DONE!"
echo "  GitHub: https://github.com/Sunshine535/nips-clox"
echo "  ZIP: /workspace/nips-clox-results.zip"
ls -lh /workspace/nips-clox-results.zip
