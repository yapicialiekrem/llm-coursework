#!/usr/bin/env bash
# End-to-end pipeline run: LoRA training → final COMET → RAG accuracy.
# Each step exits on failure (set -e) so we don't waste GPU time on cascading errors.
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence-transformers}"
export COMET_CACHE="${COMET_CACHE:-/workspace/.cache/comet}"
export PYTHONUNBUFFERED=1
# silence wandb if user not logged in
export WANDB_MODE="${WANDB_MODE:-disabled}"
# silence tokenizer fork warning
export TOKENIZERS_PARALLELISM=false

cd /workspace/llm-final-project
mkdir -p outputs/logs

START_TIME=$(date +%s)
echo "=== [$(date)] STEP 1/3: LoRA training ==="
python -m src.task1_lora_mt.train --config configs/lora_config.yaml 2>&1 | tee outputs/logs/train.log
T1=$(date +%s)
echo "=== [$(date)] STEP 1 done in $((T1-START_TIME))s ==="

echo "=== [$(date)] STEP 2/3: LoRA COMET eval (skip baseline, reuse existing) ==="
python -m src.task1_lora_mt.evaluate_comet \
    --config configs/lora_config.yaml \
    --skip-baseline \
    --output outputs/logs/comet_lora.json \
    2>&1 | tee outputs/logs/comet_eval.log
T2=$(date +%s)
echo "=== [$(date)] STEP 2 done in $((T2-T1))s ==="

echo "=== [$(date)] STEP 3/3: RAG accuracy (zero-shot vs RAG) ==="
python -m src.task2_rag.evaluate_accuracy --config configs/rag_config.yaml 2>&1 | tee outputs/logs/rag_eval.log
T3=$(date +%s)
echo "=== [$(date)] STEP 3 done in $((T3-T2))s ==="

END_TIME=$(date +%s)
echo "=== [$(date)] PIPELINE COMPLETE in $((END_TIME-START_TIME))s ==="
echo "Outputs:"
echo "  - outputs/checkpoints/qwen25_7b_lora_wmt16/  (LoRA adapter)"
echo "  - outputs/logs/comet_baseline.json           (baseline scores)"
echo "  - outputs/logs/comet_lora.json               (LoRA scores)"
echo "  - outputs/logs/rag_eval_results.json         (RAG accuracy)"
