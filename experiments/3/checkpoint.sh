#!/bin/bash
# sudo sysctl kernel.io_uring_disabled=2
set -euo pipefail

DUMP_DIR=${1:-dump-$(date +%s)}
LOGS_DIR="/home/joel/Projects/fast-vllm/logs"
CHECKPOINTS_DIR="/home/joel/Projects/fast-vllm/experiments/2/checkpoints"
SHM_DIR="$HOME/Projects/fast-vllm/shm"

mkdir -p "$LOGS_DIR"
mkdir -p "${CHECKPOINTS_DIR}/${DUMP_DIR}"

rm -rf "$SHM_DIR"
mkdir -p "$SHM_DIR"
chmod 1777 "$SHM_DIR"

LOG="${LOGS_DIR}/${DUMP_DIR}.log"

docker rm -f fast-vllm 2>/dev/null || true

echo "Starting vLLM container... (server log: $LOG)"
docker run --rm --init --gpus all --name fast-vllm \
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
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e TORCH_NUM_THREADS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  vllm-dev \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --enable-sleep-mode \
  > "$LOG" 2>&1 &
DOCKER_PID=$!

echo "Waiting for /v1/models..."
until curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null | grep -q 200; do
  sleep 0.5
  kill -0 $DOCKER_PID 2>/dev/null || { echo "Container died before server up"; cat "$LOG"; exit 1; }
done

echo "Server up. Sending warmup request..."
PROMPT='{"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "Hello", "max_tokens": 1}'
until curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" -d "$PROMPT" 2>/dev/null | grep -q 200; do
  sleep 0.5
  kill -0 $DOCKER_PID 2>/dev/null || { echo "Container died during warmup"; exit 1; }
done

echo "Putting vLLM to sleep..."
curl -s -X POST http://localhost:8000/sleep -H "Content-Type: application/json" -d '{"level": 2}'
echo ""

SLEEP_STATE=$(curl -s http://localhost:8000/is_sleeping)
echo "Sleep state: $SLEEP_STATE"

echo "Warmed and asleep. Dumping..."

docker exec fast-vllm bash -c '
  # Remove POSIX semaphores - chrek pattern
  rm -f /dev/shm/sem.* 2>/dev/null || true
  ls /dev/shm/
'
docker exec \
  -e DUMP_DIR="$DUMP_DIR" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  fast-vllm bash -c '
  set -euo pipefail

  PID=$(pgrep -fo "python3.*vllm serve")

  echo "Running criu dump..."
  set +e
  criu dump --tree $PID \
    -L /usr/local/lib/criu \
    --images-dir /checkpoints/$DUMP_DIR \
    --tcp-established \
    --shell-job \
    --ext-unix-sk \
    --skip-in-flight \
    --link-remap \
    --ghost-limit 100M \
    --allow-uprobes \
    --enable-external-masters \
    -v4 \
    -o criu.log
  CRIU_EXIT=$?
  set -e

  echo "=== criu.log tail ==="
  tail -40 /checkpoints/$DUMP_DIR/criu.log

  chown -R "$HOST_UID:$HOST_GID" /checkpoints/$DUMP_DIR
  exit $CRIU_EXIT
'
DUMP_EXIT=$?

if [ $DUMP_EXIT -eq 0 ]; then
  echo "Dump succeeded. Images in ${CHECKPOINTS_DIR}/${DUMP_DIR}/"
  du -sh "${CHECKPOINTS_DIR}/${DUMP_DIR}/"
  echo "Shared shm dir: $SHM_DIR (keep this for restore)"
  echo "Dump dir name: ${DUMP_DIR}"
else
  echo "Dump failed (exit $DUMP_EXIT). Full log: ${CHECKPOINTS_DIR}/${DUMP_DIR}/criu.log"
fi

docker rm -f fast-vllm 2>/dev/null || true
wait $DOCKER_PID 2>/dev/null || true
