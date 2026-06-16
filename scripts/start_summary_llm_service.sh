#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${WHISPERLIVE_APP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$APP_DIR"

MODEL_PATH="${SUMMARY_MODEL_PATH:-model/LLM/Qwen3-4B-AWQ}"
MODEL_NAME="${SUMMARY_MODEL_NAME:-qwen3-4b-awq}"
HOST="${SUMMARY_HOST:-127.0.0.1}"
PORT="${SUMMARY_PORT:-8001}"
MAX_MODEL_LEN="${SUMMARY_MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${SUMMARY_GPU_MEMORY_UTILIZATION:-0.70}"
QUANTIZATION="${SUMMARY_QUANTIZATION:-awq_marlin}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --quantization "$QUANTIZATION" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
