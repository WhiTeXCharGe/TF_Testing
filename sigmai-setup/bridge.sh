#!/usr/bin/env bash
# bridge.sh — the per-run Batch task for Sigma-I (engine=sigmai).
#
# One run = one bridge task on a pool-sigmai node whose Sigma-I stack is already up
# (start-sigmai.sh). It does NOT solve anything itself; it is an HTTP client:
#   1. POST /jobs          (EnvConfig.yaml + Schedule.yaml + options from sigmai_options.json)
#   2. poll GET /jobs/{id}/status every POLL_INTERVAL_SEC → write status/{runId}.json
#      (the API controller reads that file live through the Batch task-file API)
#   3. on finish: download result_Schedule.yaml, result.json, logs → output/…
#
# Layout (Batch defaults; override the *_DIR vars to run it on a laptop):
#   in : $BRIDGE_INPUT_DIR/{EnvConfig.yaml, Schedule.yaml, sigmai_options.json}
#        default $AZ_BATCH_TASK_WORKING_DIR/input/input/$RUN_ID
#   out: $BRIDGE_WORK_DIR/output/result_Schedule.yaml
#        $BRIDGE_WORK_DIR/output/sigmai/*          (result.json, logs, bridge.log)
#        $BRIDGE_WORK_DIR/status/$RUN_ID.json      (live + final status)
#
# Exit codes: 0 Completed · 1 Failed · 143 Cancelled (SIGTERM from a Batch terminate).
# Status schema: documents/SigmaI/SigmaI_Azure_DetailedDesign20261013.md §9.
set -uo pipefail

: "${RUN_ID:?RUN_ID is required}"
PY=""
for c in python3 python; do   # first one that actually runs (Windows has a non-working python3 stub)
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "python3 not found"; exit 1; }

WORK="${BRIDGE_WORK_DIR:-${AZ_BATCH_TASK_WORKING_DIR:-$PWD}}"
INPUT="${BRIDGE_INPUT_DIR:-$WORK/input/input/$RUN_ID}"
SIGMAI_HOME="${SIGMAI_HOME:-${AZ_BATCH_NODE_SHARED_DIR:-/mnt/sigmai-shared}/sigmai}"
DATA_DIR="${SIGMAI_DATA_DIR:-$SIGMAI_HOME/data}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-10}"
CANCEL_WAIT_SEC="${CANCEL_WAIT_SEC:-45}"
DEADLINE_MARGIN_SEC="${DEADLINE_MARGIN_SEC:-600}"
MAX_STATUS_FAILURES="${MAX_STATUS_FAILURES:-6}"
NODE_ID="${AZ_BATCH_NODE_ID:-local}"

OUT="$WORK/output"; SGOUT="$OUT/sigmai"; STATUS_DIR="$WORK/status"
STATUS_FILE="$STATUS_DIR/$RUN_ID.json"
mkdir -p "$SGOUT" "$STATUS_DIR"

# Everything below is also kept in output/sigmai/bridge.log (uploaded even on failure).
exec > >(tee -a "$SGOUT/bridge.log") 2>&1
log() { echo "[bridge $(date -u +%H:%M:%S)] $*"; }

# ── Sigma-I endpoint + key ───────────────────────────────────────────────
if [ -z "${SIGMAI_API_URL:-}" ] || [ -z "${SIGMAI_API_KEY:-}" ]; then
  [ -f "$SIGMAI_HOME/.env.client" ] || { log "no $SIGMAI_HOME/.env.client (did start-sigmai.sh run on this node?)"; }
  # shellcheck disable=SC1091
  [ -f "$SIGMAI_HOME/.env.client" ] && . "$SIGMAI_HOME/.env.client"
  SIGMAI_API_URL="${SIGMAI_API_URL:-${API_BASE_URL:-http://127.0.0.1:8000}}"
  SIGMAI_API_KEY="${SIGMAI_API_KEY:-${API_KEY:-}}"
fi
API="${SIGMAI_API_URL%/}"
AUTH=(-H "X-API-Key: ${SIGMAI_API_KEY}")

# ── status.json writer (atomic: tmp + mv, so the live read never sees half a file) ───
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
OPTION_NAME="sigmai"
SOLVER_JSON="null"
SUMMARY_JSON="null"
JOB_STATUS=""

write_status() {  # status progress [err_type err_message]
  STATUS="$1" PROGRESS="${2:-0}" ERR_TYPE="${3:-}" ERR_MSG="${4:-}" \
  RUN_ID="$RUN_ID" STARTED_AT="$STARTED_AT" NODE_ID="$NODE_ID" OPTION_NAME="$OPTION_NAME" \
  SOLVER_JSON="$SOLVER_JSON" SUMMARY_JSON="$SUMMARY_JSON" JOB_STATUS="$JOB_STATUS" \
  STATUS_FILE="$STATUS_FILE" "$PY" - <<'PYEOF'
import json, os, datetime
e = os.environ
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
st = e["STATUS"]
terminal = st in ("Completed", "Failed", "Cancelled")
try: progress = float(e["PROGRESS"])
except ValueError: progress = 0
doc = {
    "runId": e["RUN_ID"], "status": st, "stage": None,
    "progress": round(progress, 1),
    "startedAt": e["STARTED_AT"], "updatedAt": now,
    "finishedAt": now if terminal else None,
    "error": {"type": e["ERR_TYPE"], "message": e["ERR_MSG"]} if e["ERR_TYPE"] else None,
    "output": f"output/{e['RUN_ID']}/result_Schedule.yaml" if st == "Completed" else None,
    "engine": "sigmai", "option": e["OPTION_NAME"], "nodeId": e["NODE_ID"],
    "solver": json.loads(e["SOLVER_JSON"]),
    "sigmai": {"jobStatus": e["JOB_STATUS"] or None, "summary": json.loads(e["SUMMARY_JSON"])},
}
tmp = e["STATUS_FILE"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as f: json.dump(doc, f, indent=2)
os.replace(tmp, e["STATUS_FILE"])
PYEOF
}

fail() {  # err_type message
  log "FAILED: $1 — $2"
  collect_artifacts
  write_status Failed 0 "$1" "$2"
  exit 1
}

# ── copy logs/result summary into output/sigmai (best effort) ────────────
collect_artifacts() {
  local jd="$DATA_DIR/jobs/$RUN_ID" f
  curl -sS --max-time 30 "${AUTH[@]}" -o "$SGOUT/result.json" "$API/jobs/$RUN_ID/result" >/dev/null 2>&1 \
    || rm -f "$SGOUT/result.json"
  for f in opt_output/result_summary.json opt_output/optimization.log logs/worker.log logs/error.log; do
    [ -e "$jd/$f" ] || continue
    cp -f "$jd/$f" "$SGOUT/$(basename "$f")" 2>/dev/null \
      || sudo -n cat "$jd/$f" > "$SGOUT/$(basename "$f")" 2>/dev/null || true
  done
}

# curl in Git Bash for Windows is a native build: give it C:/... style paths (no-op on Linux).
cpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf "%s" "$1"; fi; }

# ── inputs ───────────────────────────────────────────────────────────────
log "run=$RUN_ID node=$NODE_ID api=$API input=$INPUT"
for f in EnvConfig.yaml Schedule.yaml sigmai_options.json; do
  [ -f "$INPUT/$f" ] || fail InvalidInput "missing input file: $INPUT/$f"
done

FORM_FILE="$WORK/.form_fields.txt"
prep="$(INPUT="$INPUT" RUN_ID="$RUN_ID" FORM_FILE="$FORM_FILE" DEADLINE_MARGIN_SEC="$DEADLINE_MARGIN_SEC" "$PY" - <<'PYEOF'
import json, os, shlex
e = os.environ
o = json.load(open(os.path.join(e["INPUT"], "sigmai_options.json"), encoding="utf-8"))
fields = [("job_id", e["RUN_ID"])]
if o.get("scenarioName"): fields.append(("scenario_name", str(o["scenarioName"]).replace("\n", " ")))
fields += [("stage1_solver", o["stage1Solver"]), ("stage2_solver", o["stage2Solver"]),
           ("stage1_timeout", str(o["stage1TimeoutSec"])), ("stage2_timeout", str(o["stage2TimeoutSec"]))]
if o.get("stage1NumSolutionsLimit"): fields.append(("stage1_num_solutions_limit", str(o["stage1NumSolutionsLimit"])))
if o.get("stage2NumSolutionsLimit"): fields.append(("stage2_num_solutions_limit", str(o["stage2NumSolutionsLimit"])))
if o.get("weightPreset"): fields.append(("weight_preset", o["weightPreset"]))
if o.get("disabledConstraints"): fields.append(("disabled_constraints", json.dumps(o["disabledConstraints"])))
with open(e["FORM_FILE"], "w", encoding="utf-8", newline="\n") as f:
    for k, v in fields: f.write(f"{k}={v}\n")
solver = {k: o.get(k) for k in ("stage1Solver", "stage2Solver", "stage1TimeoutSec", "stage2TimeoutSec")}
print("OPTION_NAME=" + shlex.quote(f"sigmai-{o['stage1Solver']}-{o['stage2Solver']}"))
print("SOLVER_JSON=" + shlex.quote(json.dumps({"stage1": o["stage1Solver"], "stage2": o["stage2Solver"],
      "stage1TimeoutSec": o["stage1TimeoutSec"], "stage2TimeoutSec": o["stage2TimeoutSec"]})))
print("BRIDGE_DEADLINE_SEC=" + str(int(o["stage1TimeoutSec"]) + int(o["stage2TimeoutSec"]) + int(e["DEADLINE_MARGIN_SEC"])))
PYEOF
)" || fail InvalidInput "could not read sigmai_options.json"
eval "$prep"

FORM_ARGS=()
while IFS= read -r line; do line="${line%$'\r'}"; [ -n "$line" ] && FORM_ARGS+=(--form-string "$line"); done < "$FORM_FILE"
log "option=$OPTION_NAME deadline=${BRIDGE_DEADLINE_SEC}s"

# ── cancel (Batch terminate → SIGTERM) ───────────────────────────────────
CANCELLING=false
cancel() {
  $CANCELLING && return; CANCELLING=true
  log "SIGTERM received → cancelling Sigma-I job $RUN_ID"
  curl -sS --max-time 15 -X DELETE "${AUTH[@]}" "$API/jobs/$RUN_ID" >/dev/null 2>&1 || true
  local end=$(( $(date +%s) + CANCEL_WAIT_SEC )) s
  while [ "$(date +%s)" -lt "$end" ]; do
    s="$(curl -sS --max-time 10 "${AUTH[@]}" "$API/jobs/$RUN_ID/status" 2>/dev/null | parse_status | cut -d'|' -f1)"
    [ "$s" = "canceled" ] || [ "$s" = "finished" ] || [ "$s" = "failed" ] && break
    sleep 2
  done
  collect_artifacts
  JOB_STATUS="${s:-cancel_requested}"
  write_status Cancelled 0
  exit 143
}
trap cancel TERM INT

# "status|progress|error_type|error_message" from a Sigma-I status response on stdin
parse_status() {
  "$PY" -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: print("|||"); raise SystemExit
st = (d.get("job_status") or {}).get("status", "")
pr = (d.get("progress") or {}).get("estimated_progress_rate", 0)
er = d.get("error") or {}
print("%s|%s|%s|%s" % (st, pr, er.get("type") or "", (er.get("message") or "").replace("|", "/").replace("\n", " ")))
' | tr -d '\r'
}

# ── 1. submit ────────────────────────────────────────────────────────────
write_status Running 0
log "POST /jobs"
submitted=false
for attempt in 1 2 3; do
  code="$(curl -sS --max-time 120 -o "$WORK/.post_resp.json" -w '%{http_code}' -X POST "$API/jobs" "${AUTH[@]}" \
      "${FORM_ARGS[@]}" \
      -F "environment=@$(cpath "$INPUT/EnvConfig.yaml");type=application/x-yaml" \
      -F "schedule=@$(cpath "$INPUT/Schedule.yaml");type=application/x-yaml" 2>>"$SGOUT/bridge.log")" || code=000
  body="$(cat "$WORK/.post_resp.json" 2>/dev/null || true)"
  if [ "$code" = "202" ]; then submitted=true; break; fi
  if echo "$body" | grep -q 'JOB_ALREADY_EXISTS'; then
    log "job already exists (retry after a node/task restart) → resume polling"; submitted=true; break
  fi
  case "$code" in
    4*) fail InvalidInput "Sigma-I rejected the job (HTTP $code): $(echo "$body" | head -c 400)";;
  esac
  log "POST failed (HTTP $code), attempt $attempt/3"; sleep $(( attempt * 5 ))
done
$submitted || fail SigmaIUnavailable "could not reach Sigma-I at $API after 3 attempts"
log "job accepted"

# ── 2. poll ──────────────────────────────────────────────────────────────
t0="$(date +%s)"; failures=0
while true; do
  resp="$(curl -sS --max-time 30 "${AUTH[@]}" "$API/jobs/$RUN_ID/status" 2>/dev/null)"; rc=$?
  if [ $rc -ne 0 ] || [ -z "$resp" ]; then
    failures=$((failures+1)); log "status poll failed ($failures/$MAX_STATUS_FAILURES)"
    [ "$failures" -ge "$MAX_STATUS_FAILURES" ] && fail SigmaIUnavailable "lost contact with Sigma-I"
  else
    failures=0
    IFS='|' read -r JOB_STATUS progress etype emsg <<<"$(echo "$resp" | parse_status)"
    case "$JOB_STATUS" in
      queued|running|cancel_requested)
        write_status Running "${progress:-0}"
        ;;
      finished)
        write_status Running 99
        break
        ;;
      failed)
        fail "${etype:-OPTIMIZATION_FAILED}" "${emsg:-Sigma-I reported failure}"
        ;;
      canceled)
        collect_artifacts; write_status Cancelled 0; exit 143
        ;;
      *)
        log "unexpected status response: $(echo "$resp" | head -c 200)"
        ;;
    esac
  fi
  if [ $(( $(date +%s) - t0 )) -gt "$BRIDGE_DEADLINE_SEC" ]; then
    log "bridge deadline (${BRIDGE_DEADLINE_SEC}s) exceeded → deleting job"
    curl -sS --max-time 15 -X DELETE "${AUTH[@]}" "$API/jobs/$RUN_ID" >/dev/null 2>&1 || true
    fail TIMEOUT "No result within ${BRIDGE_DEADLINE_SEC}s (stage limits + margin)"
  fi
  sleep "$POLL_INTERVAL_SEC" & wait $!   # `wait` lets the SIGTERM trap fire immediately
done

# ── 3. save result ───────────────────────────────────────────────────────
log "job finished → downloading result"
curl -sS --max-time 120 -f "${AUTH[@]}" -o "$OUT/result_Schedule.yaml" "$API/jobs/$RUN_ID/download" \
  || { rm -f "$OUT/result_Schedule.yaml"; fail ResultFileNotAvailable "Sigma-I finished but result_Schedule.yaml could not be downloaded"; }
collect_artifacts

SUMMARY_JSON="$(RESULT="$SGOUT/result.json" "$PY" - <<'PYEOF'
import json, os
try: d = json.load(open(os.environ["RESULT"], encoding="utf-8"))
except Exception: print("null"); raise SystemExit
s = d.get("summary") or {}
print(json.dumps({"validationPassed": s.get("validation_passed"), "finalScore": s.get("final_score"),
                  "stage1Status": (d.get("stage1_result") or {}).get("status"),
                  "stage2Status": (d.get("stage2_result") or {}).get("status")}))
PYEOF
)"
[ -n "$SUMMARY_JSON" ] || SUMMARY_JSON="null"
if echo "$SUMMARY_JSON" | grep -q '"validationPassed": false'; then
  log "WARNING: Sigma-I finished but validation_passed=false — the schedule may be infeasible"
fi
write_status Completed 100
log "Completed"
exit 0
