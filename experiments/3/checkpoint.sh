#!/bin/bash
# sudo sysctl kernel.io_uring_disabled=2
set -euo pipefail

DUMP_DIR=${1:-dump-$(date +%s)}
LOGS_DIR="/home/joel/Projects/fast-vllm/logs"
CHECKPOINTS_DIR="/home/joel/Projects/fast-vllm/experiments/3/checkpoints"
WEIGHTS_DIR="/home/joel/Projects/fast-vllm/experiments/3/weights"
SHM_DIR="$HOME/Projects/fast-vllm/shm"

mkdir -p "$LOGS_DIR"
mkdir -p "${CHECKPOINTS_DIR}/${DUMP_DIR}"
mkdir -p "${WEIGHTS_DIR}"

rm -rf "$SHM_DIR"
mkdir -p "$SHM_DIR"
chmod 1777 "$SHM_DIR"

LOG="${LOGS_DIR}/${DUMP_DIR}.log"

docker rm -f fast-vllm 2>/dev/null || true

echo "Starting vLLM container... (server log: $LOG)"

#-v /home/joel/Projects/vllm:/opt/vllm \
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
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.triton:/root/.triton \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.nv:/root/.nv \
  -v "${SHM_DIR}:/dev/shm" \
  -v "${CHECKPOINTS_DIR}:/checkpoints" \
  -v "${LOGS_DIR}:/logs" \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e TORCH_NUM_THREADS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -v "${WEIGHTS_DIR}:/weights" \
  -e FAST_VLLM_WEIGHTS_PATH="/weights/weights" \
  --entrypoint bash \
  vllm-dev \
  -c "exec vllm serve meta-llama/Llama-3.2-3B-Instruct \
        --gpu-memory-utilization 0.80 \
        --max-model-len 8192 \
        --enable-sleep-mode \
        > /dev/null 2>&1" &
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
curl -s -X POST 'http://localhost:8000/sleep?level=2'
echo ""

SLEEP_STATE=$(curl -s http://localhost:8000/is_sleeping)
echo "Sleep state: $SLEEP_STATE"

echo "\n\nGPU mem just before dump:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "\n\nHost RSS for EngineCore:"
docker exec fast-vllm bash -c '
    for pid in $(pgrep -f vllm); do
        rss=$(grep VmRSS /proc/$pid/status 2>/dev/null | awk "{print \$2}")
        cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null | cut -c1-60)
        echo "  $pid  ${rss}kB  $cmd"
    done
'

echo "=== Memory breakdown by process ==="
docker exec fast-vllm bash -c '
    for pid in $(ls /proc | grep -E "^[0-9]+$" | sort -n); do
        if [ -r /proc/$pid/comm ]; then
            comm=$(cat /proc/$pid/comm 2>/dev/null)
            if [[ "$comm" =~ python|vllm|EngineCore ]]; then
                rss=$(grep VmRSS /proc/$pid/status | awk "{print \$2}")
                echo "PID $pid ($comm): ${rss} kB"
            fi
        fi
    done
'

docker exec fast-vllm bash -c '
    PID=$(pgrep -f EngineCore | head -1)
    awk "
        /^[0-9a-f]+-[0-9a-f]+/ {name=\$6; if (name==\"\") name=\"[anon]\"}
        /^Rss:/ {
            if (name ~ /\.so/) cat=\"libs\"
            else if (name == \"[heap]\") cat=\"heap\"
            else if (name == \"[stack]\") cat=\"stack\"
            else if (name == \"[anon]\") cat=\"anon\"
            else if (name ~ /\.safetensors|\.bin/) cat=\"weights_file\"
            else if (name ~ /\/dev\/shm/) cat=\"shm\"
            else if (name ~ /^\//) cat=\"other_file\"
            else cat=\"unknown\"
            sum[cat] += \$2
        }
        END { for (k in sum) printf \"%-15s %10d kB\n\", k, sum[k] }
    " /proc/$PID/smaps | sort -k 2 -rn
'

temp_PID=$(pgrep -f EngineCore | head -1)

docker exec fast-vllm bash -c "
pmap -x $temp_PID | sort -k3 -nr | head -40
"

docker exec fast-vllm bash -c '
PID=$(pgrep -f EngineCore | head -1)
echo "=== EngineCore PID: $PID ==="

echo ""
echo "=== Top RSS regions (pmap) ==="
echo "address    kbytes    rss    dirty    mode    mapping"
pmap -x $PID | sort -k3 -nr | head -40

echo ""
echo "=== smaps: large regions with Private_Dirty ==="
awk "
/^[0-9a-f]/{
  if (rss+0 > 50000) print rss, pd, name
  name=\$6; rss=0; pd=0
}
/^Rss:/{rss=\$2}
/^Private_Dirty:/{pd=\$2}
END { if (rss+0 > 50000) print rss, pd, name }
" /proc/$PID/smaps | sort -nr | head -50

echo ""
'

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
