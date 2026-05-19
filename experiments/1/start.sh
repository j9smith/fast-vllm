#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTS="${EXPERIMENT_DIR}/events.log"
MODE=${1:-warm}

> "$EVENTS"

docker stop fast-vllm 2>/dev/null || true
docker rm   fast-vllm 2>/dev/null || true

if [[ "$MODE" == "cold" ]]; then
  echo "Wiping compile caches..."
  sudo rm -rf ~/.cache/vllm/* ~/.triton/* ~/.cache/flashinfer/* ~/.nv/ComputeCache 2>/dev/null || true
  mkdir -p ~/.cache/vllm ~/.triton ~/.cache/flashinfer ~/.nv
  sudo sync && sudo sysctl -w vm.drop_caches=3 >/dev/null
elif [[ "$MODE" != "warm" ]]; then
  echo "Unknown mode: $MODE (use 'cold' or 'warm')"
  exit 1
fi

echo "MODE=$MODE" >> "$EVENTS"

T_START=$(date +%s%N)
echo "CONTAINER_START=$(date +%s%N)" >> "$EVENTS"

docker run --rm --gpus all --name fast-vllm \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --cap-add=CHECKPOINT_RESTORE \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --ipc=host \
  -v /home/joel/Projects/vllm:/opt/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.triton:/root/.triton \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.nv:/root/.nv \
  -v /home/joel/Projects/fast-vllm/checkpoints:/checkpoints \
  -p 8000:8000 \
  vllm-dev \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  2>&1 &
DOCKER_PID=$!

until curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; do
  sleep 0.1
  kill -0 $DOCKER_PID 2>/dev/null || { echo "Container died before server up"; exit 1; }
done
echo "SERVER_UP=$(date +%s%N)" >> "$EVENTS"

PROMPT='{"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "Hello", "max_tokens": 1}'
until curl -sf -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" -d "$PROMPT" 2>/dev/null | grep -q 200; do
  sleep 0.1
  kill -0 $DOCKER_PID 2>/dev/null || { echo "Container died before inference"; exit 1; }
done
echo "FIRST_TOKEN=$(date +%s%N)" >> "$EVENTS"

docker stop fast-vllm >/dev/null 2>&1 || true
wait $DOCKER_PID 2>/dev/null || true