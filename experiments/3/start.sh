#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTS="${EXPERIMENT_DIR}/events.log"
CHECKPOINTS_DIR="/home/joel/Projects/fast-vllm/experiments/2/checkpoints"
SHM_DIR="$HOME/Projects/fast-vllm/shm"

DUMP_DIR=${1:-}
> "$EVENTS"

if [[ -z "$DUMP_DIR" ]]; then
  DUMP_DIR=$(ls -t "$CHECKPOINTS_DIR" | head -1)
  echo "No checkpoint specified, using most recent: $DUMP_DIR"
fi

if [[ ! -d "${CHECKPOINTS_DIR}/${DUMP_DIR}" ]]; then
  echo "Dump dir not found: ${CHECKPOINTS_DIR}/${DUMP_DIR}"
  exit 1
fi

if [[ ! -d "$SHM_DIR" ]]; then
  echo "Shared shm dir missing: $SHM_DIR"
  echo "This must persist from the dump. Did you wipe /tmp?"
  exit 1
fi

echo "CHECKPOINT=${DUMP_DIR}" >> "$EVENTS"

docker rm -f fast-vllm 2>/dev/null || true

echo "Restoring from ${DUMP_DIR}..."
echo "CONTAINER_START=$(date +%s%N)" >> "$EVENTS"

docker run --rm --gpus all --name fast-vllm \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_TIME \
  --cap-add=SYS_RESOURCE \
  --cap-add=CHECKPOINT_RESTORE \
  --cap-add=NET_ADMIN \
  --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v /home/joel/Projects/vllm:/opt/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.triton:/root/.triton \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.nv:/root/.nv \
  -v "${SHM_DIR}:/dev/shm" \
  -v "${CHECKPOINTS_DIR}:/checkpoints" \
  -p 8000:8000 \
  --entrypoint bash \
  vllm-dev \
  -c "criu restore -L /usr/local/lib/criu \
      --images-dir /checkpoints/${DUMP_DIR} \
      --tcp-established \
      --shell-job \
      --enable-external-masters \
      --allow-uprobes \
      -v4 \
      -o restore.log" &
RESTORE_PID=$!

echo "RESTORE_ISSUED=$(date +%s%N)" >> "$EVENTS"

cleanup() {
  docker rm -f fast-vllm >/dev/null 2>&1 || true
  wait $RESTORE_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for /v1/models..."
DEADLINE=$(($(date +%s) + 60))
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; do
  sleep 0.1
  kill -0 $RESTORE_PID 2>/dev/null || {
    echo "Restore container died"
    sudo tail -40 "${CHECKPOINTS_DIR}/${DUMP_DIR}/restore.log" 2>/dev/null || true
    exit 1
  }
  if (( $(date +%s) > DEADLINE )); then
    echo "Timeout waiting for /v1/models"
    exit 1
  fi
done

echo "SERVER_UP=$(date +%s%N)" >> "$EVENTS"

echo "Waking vLLM..."
curl -s -X POST http://localhost:8000/wake_up >/dev/null
echo "WAKE_DONE=$(date +%s%N)" >> "$EVENTS"

echo "Sending test request..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "The capital of France is", "max_tokens": 10}')
echo "FIRST_TOKEN=$(date +%s%N)" >> "$EVENTS"

COMPLETION=$(echo "$RESPONSE" | jq -r '.choices[0].text // empty')

if [[ -z "$COMPLETION" ]]; then
  echo "ERROR: no completion in response"
  echo "Raw response: $RESPONSE"
  exit 1
fi

echo "Completion: $COMPLETION"
echo "COMPLETION=$COMPLETION" >> "$EVENTS"

echo "Done."