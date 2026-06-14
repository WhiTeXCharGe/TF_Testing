#!/usr/bin/env bash
# Container entrypoint for the Timefold solver (web/Timefold version).
#
# Flow:
#   1. Validate /work/input/EnvConfig.yaml + Schedule.yaml
#   2. Copy Schedule.yaml → /work/output/result_Schedule.yaml (solver mutates in place)
#   3. Run solver
#   4. Maintain /work/status/<RUN_ID>.json with full spec schema throughout
#
# Cancel = SIGTERM (docker stop, ACA Job stop)
#   → writes status=Cancelled, exits 143
set -euo pipefail

RUN_ID="${RUN_ID:-local-$(date +%Y%m%d_%H%M%S)}"
INPUT_DIR="${INPUT_DIR:-/work/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/output}"
STATUS_DIR="${STATUS_DIR:-/work/status}"

ENV_FILE="${INPUT_DIR}/EnvConfig.yaml"
SCHED_FILE_IN="${INPUT_DIR}/Schedule.yaml"
SCHED_FILE_OUT="${OUTPUT_DIR}/result_Schedule.yaml"
STATUS_FILE="${STATUS_DIR}/${RUN_ID}.json"

mkdir -p "${STATUS_DIR}"

# ─── helpers ─────────────────────────────────────────────────────────────────

now_iso() { date -u +%Y-%m-%dT%H:%M:%S.000Z; }
STARTED_AT="$(now_iso)"

# Atomic write: write to .tmp then rename so readers never see partial JSON.
_write_json() {
  local tmp="${STATUS_FILE}.tmp"
  cat > "${tmp}"
  mv -f "${tmp}" "${STATUS_FILE}"
}

write_running() {
  local stage="${1:-1}" progress="${2:-0}"
  _write_json <<EOF
{
  "runId":      "${RUN_ID}",
  "status":     "Running",
  "stage":      ${stage},
  "progress":   ${progress},
  "startedAt":  "${STARTED_AT}",
  "updatedAt":  "$(now_iso)",
  "finishedAt": null,
  "error":      null,
  "output":     null
}
EOF
}

write_completed() {
  _write_json <<EOF
{
  "runId":      "${RUN_ID}",
  "status":     "Completed",
  "stage":      1,
  "progress":   1,
  "startedAt":  "${STARTED_AT}",
  "updatedAt":  "$(now_iso)",
  "finishedAt": "$(now_iso)",
  "error":      null,
  "output":     "${SCHED_FILE_OUT}"
}
EOF
}

write_failed() {
  # write_failed <error_type> <message> <stage_or_null>
  local err_type="$1"
  local err_msg="$2"
  local stage="${3:-null}"
  local msg_escaped
  msg_escaped=$(printf '%s' "${err_msg}" | sed 's/\\/\\\\/g; s/"/\\"/g')
  _write_json <<EOF
{
  "runId":      "${RUN_ID}",
  "status":     "Failed",
  "stage":      ${stage},
  "progress":   0,
  "startedAt":  "${STARTED_AT}",
  "updatedAt":  "$(now_iso)",
  "finishedAt": "$(now_iso)",
  "error":      {"type": "${err_type}", "message": "${msg_escaped}"},
  "output":     null
}
EOF
}

write_cancelled() {
  local stage="${CURRENT_STAGE:-1}" progress="${CURRENT_PROGRESS:-0}"
  _write_json <<EOF
{
  "runId":      "${RUN_ID}",
  "status":     "Cancelled",
  "stage":      ${stage},
  "progress":   ${progress},
  "startedAt":  "${STARTED_AT}",
  "updatedAt":  "$(now_iso)",
  "finishedAt": "$(now_iso)",
  "error":      null,
  "output":     null
}
EOF
}

# ─── cancel handling ──────────────────────────────────────────────────────────

on_term() {
  echo "[entrypoint] caught SIGTERM — marking Cancelled" >&2
  write_cancelled
  if [[ -n "${SOLVER_PID:-}" ]] && kill -0 "${SOLVER_PID}" 2>/dev/null; then
    kill -TERM "${SOLVER_PID}" 2>/dev/null || true
  fi
  exit 143
}
trap on_term SIGTERM SIGINT

# ─── input validation ────────────────────────────────────────────────────────

if [[ ! -f "${ENV_FILE}" ]]; then
  write_failed "InvalidInputData" "EnvConfig.yaml not found at ${ENV_FILE}" "null"
  exit 2
fi
if [[ ! -f "${SCHED_FILE_IN}" ]]; then
  write_failed "InvalidInputData" "Schedule.yaml not found at ${SCHED_FILE_IN}" "null"
  exit 2
fi

# ─── prepare output copy ─────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

if ! cp -f "${SCHED_FILE_IN}" "${SCHED_FILE_OUT}"; then
  write_failed "OutputError" "Cannot create output file at ${SCHED_FILE_OUT}" "null"
  exit 3
fi

# ─── solve ───────────────────────────────────────────────────────────────────

CURRENT_STAGE=1
CURRENT_PROGRESS=0
write_running 1 0
echo "[entrypoint] solve started (runId=${RUN_ID})"

java -Xms1g -Xmx"${JVM_MAX_HEAP:-6g}" \
     -cp '/app/lib/*' \
     com.yourorg.scheduler.EmployeeSchedule \
     "${ENV_FILE}" \
     "${SCHED_FILE_OUT}" &
SOLVER_PID=$!
wait "${SOLVER_PID}"
SOLVER_EXIT=$?

if [[ "${SOLVER_EXIT}" -ne 0 ]]; then
  write_failed "SolverError" "solver exited with code ${SOLVER_EXIT}" "1"
  exit "${SOLVER_EXIT}"
fi

# ─── verify output ───────────────────────────────────────────────────────────

if [[ ! -f "${SCHED_FILE_OUT}" ]]; then
  write_failed "OutputError" "Solver finished but result_Schedule.yaml is missing at ${SCHED_FILE_OUT}" "1"
  exit 4
fi

write_completed
echo "[entrypoint] done — result at ${SCHED_FILE_OUT}"
