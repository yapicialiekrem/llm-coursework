#!/usr/bin/env bash
# One-shot snapshot of the running pipeline. Safe to call any time.
# Usage:
#   bash scripts/status.sh             (on the pod)
#   ssh ... 'bash /workspace/llm-final-project/scripts/status.sh'  (from local)
set -u

cd "$(dirname "$0")/.."

LOG="outputs/logs/pipeline_run.log"
echo "===================== Pipeline Status ====================="
echo "Time: $(date)"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[no nvidia-smi available]"
else
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
fi
echo

echo "--- tmux sessions ---"
tmux ls 2>/dev/null || echo "(no tmux sessions)"
echo

if [ ! -f "$LOG" ]; then
    echo "Log file not found yet: $LOG"
    exit 0
fi

echo "--- current STEP ---"
grep -E "STEP [0-9]+/3" "$LOG" | tail -3
echo

echo "--- training progress (last reported) ---"
tail -1 "$LOG" | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[.*\]' | tail -1
echo

echo "--- recent loss values (last 5) ---"
grep -oE "'loss': [0-9.]+|'eval_loss': [0-9.]+" "$LOG" | tail -5
echo

echo "--- checkpoints saved ---"
ls -1 outputs/checkpoints/qwen25_7b_lora_wmt16/ 2>/dev/null | grep -E '^(adapter_|checkpoint-)' | head -10 || echo "(none yet)"
echo

echo "--- JSON results so far ---"
for f in outputs/logs/comet_baseline.json outputs/logs/comet_lora.json outputs/logs/rag_eval_results.json; do
    if [ -f "$f" ]; then
        echo "$f:"
        cat "$f" | head -20 | sed 's/^/  /'
    else
        echo "$f: (not yet)"
    fi
done

echo
echo "--- pipeline done marker? ---"
if grep -q "PIPELINE_DONE\|PIPELINE COMPLETE" "$LOG"; then
    echo "✅ DONE — pipeline finished. Build the final report:"
    echo "   python scripts/build_final_report.py"
else
    echo "🔄 still running"
fi
echo "==========================================================="
