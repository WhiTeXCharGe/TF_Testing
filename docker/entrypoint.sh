#!/usr/bin/env bash
# Container entrypoint for the Timefold solver (web/Timefold version).
#
# Flow:
#   1. Read /work/input/EnvConfig.yaml + /work/input/Schedule.yaml
#   2. Copy Schedule.yaml to /work/output/result_Schedule.yaml (solver mutates
#      its argument in place — we work on the COPY so the input stays pristine)
#   3. Run the solver against (env, output-copy)
#   4. Maintain /work/status/<RUN_ID>.json throughout (Submitted → Running → Completed/Failed/Cancelled)
#
# Cancel = SIGTERM (e.g. `docker stop` or ACA Job execution stop)
#   → handler writes status = Cancelled and exits 143.
set -euo pipefail

RUN_ID="${RUN_ID:-local-$(date +%Y%m%d_%H%M%S)}"
INPUT_DIR="${INPUT_DIR:-/work/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/output}"
STATUS_DIR="${STATUS_DIR:-/work/status}"

ENV_FILE="${INPUT_DIR}/EnvConfig.yaml"
SCHED_FILE_IN="${INPUT_DIR}/Schedule.yaml"
SCHED_FILE_OUT="${OUTPUT_DIR}/result_Schedule.yaml"
STATUS_FILE="${STATUS_DIR}/${RUN_ID}.json"

mkdir -p "${OUTPUT_DIR}" "${STATUS_DIR}"

# ─── status helpers ────────────────────────────────────────────────────────
now_iso() { date -u +%Y-%m-%dT%H:%M:%S.000Z; }

write_status() {
  # write_status <state> [error_message]
  local state="$1"
  local err="${2:-}"
  local err_json="null"
  if [[ -n "${err}" ]]; then
    err_escaped=$(printf '%s' "${err}" | sed 's/\\/\\\\/g; s/"/\\"/g')
    err_json="{\"message\":\"${err_escaped}\"}"
  fi
  cat > "${STATUS_FILE}" <<EOF
{
  "runId":     "${RUN_ID}",
  "status":    "${state}",
  "startedAt": "${STARTED_AT}",
  "updatedAt": "$(now_iso)",
  "error":     ${err_json},
  "output":    "${SCHED_FILE_OUT}"
}
EOF
}

STARTED_AT="$(now_iso)"
write_status "Submitted"

# ─── cancel handling ───────────────────────────────────────────────────────
on_term() {
  echo "[entrypoint] caught signal — marking Cancelled" >&2
  write_status "Cancelled"
  # 143 = 128 + SIGTERM(15). ACA / k8s expect non-zero on cancel.
  if [[ -n "${SOLVER_PID:-}" ]] && kill -0 "${SOLVER_PID}" 2>/dev/null; then
    kill -TERM "${SOLVER_PID}" 2>/dev/null || true
  fi
  exit 143
}
trap on_term SIGTERM SIGINT

# ─── input validation ──────────────────────────────────────────────────────
if [[ ! -f "${ENV_FILE}" ]]; then
  write_status "Failed" "EnvConfig.yaml not found at ${ENV_FILE}"
  exit 2
fi
if [[ ! -f "${SCHED_FILE_IN}" ]]; then
  write_status "Failed" "Schedule.yaml not found at ${SCHED_FILE_IN}"
  exit 2
fi

# Work on a copy so input stays pristine.
cp -f "${SCHED_FILE_IN}" "${SCHED_FILE_OUT}"

# ─── run the solver ────────────────────────────────────────────────────────
write_status "Running"

# Run in background so the trap above can react to SIGTERM immediately
# (otherwise bash waits for the foreground process to return first).
java -Xms1g -Xmx"${JVM_MAX_HEAP:-6g}" \
     -cp '/app/lib/*' \
     com.yourorg.scheduler.EmployeeSchedule \
     "${ENV_FILE}" \
     "${SCHED_FILE_OUT}" &
SOLVER_PID=$!
wait "${SOLVER_PID}"
SOLVER_EXIT=$?

if [[ "${SOLVER_EXIT}" -ne 0 ]]; then
  write_status "Failed" "solver exited with code ${SOLVER_EXIT}"
  exit "${SOLVER_EXIT}"
fi

write_status "Completed"
echo "[entrypoint] done. result at ${SCHED_FILE_OUT}"
