#!/usr/bin/env bash
set -euo pipefail

GPUS=(0 1)
PORTS=(9090 9091)

MODEL_PATH="model/asr/small"
MAX_CLIENTS=12
MAX_CONNECTION_TIME=600
BATCH_MAX_SIZE=8
BATCH_WINDOW_MS=50
LOG_DIR="logs"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  echo "Usage: $0 {start|stop|restart|status}"
}

repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/.." && pwd
}

pid_file() {
  local gpu="$1"
  local port="$2"
  echo "${LOG_DIR}/whisperlive-gpu${gpu}-port${port}.pid"
}

log_file() {
  local gpu="$1"
  local port="$2"
  echo "${LOG_DIR}/whisperlive-gpu${gpu}-port${port}.log"
}

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

port_in_use() {
  local port="$1"

  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :${port} )" 2>/dev/null | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'
    return
  fi

  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi

  return 1
}

validate_config() {
  if [[ ${#GPUS[@]} -ne ${#PORTS[@]} ]]; then
    echo "ERROR: GPUS and PORTS must have the same length." >&2
    exit 1
  fi

  if [[ ! -f "run_server.py" ]]; then
    echo "ERROR: run_server.py not found. Run this script from the repository or scripts directory." >&2
    exit 1
  fi

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model path not found: ${MODEL_PATH}" >&2
    echo "Download it first, for example: python3 scripts/download_models.py --model small" >&2
    exit 1
  fi

  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
}

start_instance() {
  local gpu="$1"
  local port="$2"
  local pid_path
  local log_path

  pid_path="$(pid_file "${gpu}" "${port}")"
  log_path="$(log_file "${gpu}" "${port}")"

  if [[ -f "${pid_path}" ]]; then
    local existing_pid
    existing_pid="$(cat "${pid_path}")"
    if is_pid_running "${existing_pid}"; then
      echo "SKIP: GPU ${gpu}, port ${port} already running with PID ${existing_pid}."
      return
    fi
    rm -f "${pid_path}"
  fi

  if port_in_use "${port}"; then
    echo "ERROR: port ${port} is already in use. Not starting GPU ${gpu} instance." >&2
    return 1
  fi

  echo "START: GPU ${gpu}, port ${port}, log ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON_BIN}" run_server.py \
    --port "${port}" \
    --backend faster_whisper \
    --max_clients "${MAX_CLIENTS}" \
    --max_connection_time "${MAX_CONNECTION_TIME}" \
    --asr_device_index "${ASR_DEVICE_INDEX:-0}" \
    --translation_device "${TRANSLATION_DEVICE:-cpu}" \
    --batch_max_size "${BATCH_MAX_SIZE}" \
    --batch_window_ms "${BATCH_WINDOW_MS}" \
    -fw "${MODEL_PATH}" \
    > "${log_path}" 2>&1 &

  echo "$!" > "${pid_path}"
  echo "PID: $(cat "${pid_path}")"
}

start_all() {
  validate_config
  mkdir -p "${LOG_DIR}"

  local i
  for i in "${!GPUS[@]}"; do
    start_instance "${GPUS[$i]}" "${PORTS[$i]}"
  done
}

stop_instance() {
  local gpu="$1"
  local port="$2"
  local pid_path

  pid_path="$(pid_file "${gpu}" "${port}")"

  if [[ ! -f "${pid_path}" ]]; then
    echo "STOP: GPU ${gpu}, port ${port} has no PID file."
    return
  fi

  local pid
  pid="$(cat "${pid_path}")"

  if ! is_pid_running "${pid}"; then
    echo "STOP: GPU ${gpu}, port ${port} PID ${pid} is not running."
    rm -f "${pid_path}"
    return
  fi

  echo "STOP: GPU ${gpu}, port ${port}, PID ${pid}"
  kill "${pid}"

  local waited=0
  while is_pid_running "${pid}" && [[ ${waited} -lt 15 ]]; do
    sleep 1
    waited=$((waited + 1))
  done

  if is_pid_running "${pid}"; then
    echo "WARN: PID ${pid} is still running after ${waited}s. Check manually before forcing shutdown." >&2
    return 1
  fi

  rm -f "${pid_path}"
}

stop_all() {
  mkdir -p "${LOG_DIR}"

  local i
  for i in "${!GPUS[@]}"; do
    stop_instance "${GPUS[$i]}" "${PORTS[$i]}"
  done
}

status_instance() {
  local gpu="$1"
  local port="$2"
  local pid_path
  local pid_status="not running"
  local port_status="not listening"

  pid_path="$(pid_file "${gpu}" "${port}")"

  if [[ -f "${pid_path}" ]]; then
    local pid
    pid="$(cat "${pid_path}")"
    if is_pid_running "${pid}"; then
      pid_status="running, pid ${pid}"
    else
      pid_status="stale pid file, pid ${pid}"
    fi
  fi

  if port_in_use "${port}"; then
    port_status="listening"
  fi

  echo "GPU ${gpu}, port ${port}: ${pid_status}; port ${port_status}"
}

status_all() {
  local i
  for i in "${!GPUS[@]}"; do
    status_instance "${GPUS[$i]}" "${PORTS[$i]}"
  done
}

main() {
  local root
  root="$(repo_root)"
  cd "${root}"

  local command="${1:-}"
  case "${command}" in
    start)
      start_all
      ;;
    stop)
      stop_all
      ;;
    restart)
      stop_all
      start_all
      ;;
    status)
      status_all
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
