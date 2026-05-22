#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "========================================================="
echo "🚀 STARTING MULTIMODAL PROMPTOMATIX + VQA IMAGE POOL PIPELINE"
echo "========================================================="

mkdir -p "$REPO_DIR/outputs"

# Runtime configuration (override via env)
export VLLM_DEVICE="${VLLM_DEVICE:-cuda}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export VLLM_HOST="127.0.0.1"
export VLLM_PORT="${VLLM_PORT:-8000}"
export IMAGE_POOL_PATH="${IMAGE_POOL_PATH:-}"
export IMAGE_POOL_DATASET="${IMAGE_POOL_DATASET:-coco/2014}"
export SYNTHETIC_VQA_SIZE="${SYNTHETIC_VQA_SIZE:-50}"
export OPTIMIZER_API_KEY="local"
export OPTIMIZER_MODEL="openai/Qwen/Qwen2-VL-7B-Instruct"
export OPTIMIZER_API_BASE="http://127.0.0.1:${VLLM_PORT}"
export OPTIMIZER_PROVIDER="local"
export LOCAL_VLM_URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"
export OUTPUT_DIR="$REPO_DIR/outputs"

echo "📦 Checking vLLM installation..."
python3 -c "import vllm; print(f'✅ vLLM version: {vllm.__version__}')" || {
  echo "⚠️  vLLM not found in the Python environment. Install it before running." >&2
  exit 1
}

# 1. Start vLLM server in the background
echo "⏳ Launching vLLM server (${VLLM_MODEL}) on ${VLLM_HOST}:${VLLM_PORT}..."
python3 -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --device "${VLLM_DEVICE}" \
  --max-model-len 8192 \
  --trust-remote-code \
  > "$REPO_DIR/vllm_server.log" 2>&1 &
VLM_PID=$!

echo "VLM Server PID: ${VLM_PID}"

cleanup() {
  echo "🧹 Cleaning up vLLM server (PID: ${VLM_PID})..."
  kill "${VLM_PID}" 2>/dev/null || true
  wait "${VLM_PID}" 2>/dev/null || true
  echo "✅ Cleanup complete"
}
trap cleanup EXIT

wait_for_port() {
  local port="$1"
  local label="$2"
  local timeout="${3:-300}"
  local start
  start=$(date +%s)

  printf "⏳ Waiting for %s on port %s...\n" "$label" "$port"
  while true; do
    if python3 -c "import socket,sys; port=${port}; try: socket.create_connection(('127.0.0.1', port), timeout=2); except Exception: sys.exit(1)" 2>/dev/null; then
      printf "✅ %s is online!\n" "$label"
      return 0
    fi

    local now
    now=$(date +%s)
    if (( now - start > timeout )); then
      printf "❌ Timeout waiting for %s on port %s\n" "$label" "$port"
      return 1
    fi
    sleep 5
  done
}

if ! wait_for_port "$VLLM_PORT" "vLLM Server" 420; then
  exit 1
fi

echo "========================================================="
echo "🎉 vLLM server online. Running integrated VQA pipeline..."
echo "========================================================="

# 2. Run the integration example for VQA image pool + synthetic generation
python3 <<PY
import os
from pathlib import Path
import sys

repo_dir = Path(os.getcwd())
sys.path.insert(0, str(repo_dir))

from integration_example import PromptomatixVQAConfig, IntegratedPromptomatixPipeline

print('📦 Initializing the Promptomatix VQA integration pipeline...')

config = PromptomatixVQAConfig()
config.image_pool_path = os.environ.get('IMAGE_POOL_PATH') or None
config.image_pool_dataset = os.environ.get('IMAGE_POOL_DATASET')
config.vlm_base_url = os.environ.get('OPTIMIZER_API_BASE', f'http://127.0.0.1:{os.environ.get("VLLM_PORT", "8000")}')
config.synthetic_vqa_size = int(os.environ.get('SYNTHETIC_VQA_SIZE', '50'))

pipeline = IntegratedPromptomatixPipeline(config)
if not pipeline.setup_components():
    raise SystemExit('Pipeline setup failed')

pipeline.generate_synthetic_vqa_dataset()

pipeline.print_summary()
PY

echo "========================================================="
echo "🎉 VQA image pool integration complete. Running Promptomatix multimodal optimization..."
echo "========================================================="

# 3. Run the Promptomatix multimodal optimization example
python3 "$REPO_DIR/promptomatix/examples/scripts/multimodal_optimization.py"

echo "========================================================="
echo "🎉 PROMPTOMATIX MULTIMODAL PIPELINE COMPLETE!"
echo "========================================================="
