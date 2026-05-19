#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:-}
if [[ -z "$EXPERIMENT" ]]; then
  echo "Usage: ./run.sh <experiment_number> [args...]"
  exit 1
fi
shift

EXPERIMENT_DIR="experiments/${EXPERIMENT}"
START_SCRIPT="${EXPERIMENT_DIR}/start.sh"
EVENTS_FILE="${EXPERIMENT_DIR}/events.log"
LOG_DIR="${EXPERIMENT_DIR}/logs"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%dT%H%M%S).log"

if [[ ! -f "$START_SCRIPT" ]]; then
  echo "Error: ${START_SCRIPT} not found"
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "=== fast-vllm | experiment ${EXPERIMENT} ==="
echo "Log: ${LOG_FILE}"
echo

WALL_START=$(date +%s%N)
bash "$START_SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"
START_EXIT=${PIPESTATUS[0]}
WALL_END=$(date +%s%N)
WALL_MS=$(( (WALL_END - WALL_START) / 1000000 ))

echo
echo "=== results ==="
if [[ -f "$EVENTS_FILE" ]]; then
  declare -A events
  while IFS='=' read -r key val; do
    events["$key"]="$val"
  done < "$EVENTS_FILE"

  BASE=${events[CONTAINER_START]:-$WALL_START}
  ns_to_ms() { echo $(( ($1 - BASE) / 1000000 )); }

[[ -n "${events[MODE]:-}"           ]] && echo "mode              : ${events[MODE]}"
  [[ -n "${events[CHECKPOINT]:-}"     ]] && echo "checkpoint        : ${events[CHECKPOINT]}"
  [[ -n "${events[RESTORE_ISSUED]:-}" ]] && printf "restore issued    : +%dms\n" "$(ns_to_ms "${events[RESTORE_ISSUED]}")"
  [[ -n "${events[HTTP_UP]:-}"        ]] && printf "http listening    : +%dms\n" "$(ns_to_ms "${events[HTTP_UP]}")"
  [[ -n "${events[WAKE_DONE]:-}"      ]] && printf "wake_up returned  : +%dms\n" "$(ns_to_ms "${events[WAKE_DONE]}")"
  [[ -n "${events[SERVER_UP]:-}"      ]] && printf "server healthy    : +%dms\n" "$(ns_to_ms "${events[SERVER_UP]}")"
  [[ -n "${events[FIRST_TOKEN]:-}"    ]] && printf "first token       : +%dms\n" "$(ns_to_ms "${events[FIRST_TOKEN]}")"
fi

echo
printf "wall time         : %dms\n" "$WALL_MS"

exit $START_EXIT
