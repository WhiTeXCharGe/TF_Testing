#!/usr/bin/env bash
# Run input variants through a Sigma-I API and collect the results — to be run on the PC where Sigma-I runs.
#
# usage:  bash run_variants.sh <variants_dir> <sigmai_compose_dir> [variant_name ...]
#   <variants_dir>        output of make_variants.py (contains v0_baseline/, v1_.../ ...)
#   <sigmai_compose_dir>  folder with docker-compose.yml, .env.client, data/  (the README folder)
#   variant_name ...      optional: run only these (default: all, in name order)
#
# Settings (env vars):  STAGE1_TIMEOUT (default 300)  STAGE2_TIMEOUT (default 120)  MAX_WAIT (default 1800 s per variant)
#                       API_BASE_URL / API_KEY override .env.client
# Results:  <variants_dir>/results/<variant>/{summary.txt, status.json, result.json, optimization.log, scip_S1.log, ...}
#           <variants_dir>/results/ALL_SUMMARIES.txt   <- send this file
set -uo pipefail
VAR="${1:?variants dir}"; HOME_DIR="${2:?sigmai compose dir}"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=""
for c in python3 python; do   # first one that actually runs (Windows has a non-working python3 stub)
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "python not found"; exit 1; }
[ -f "$HOME_DIR/.env.client" ] && . "$HOME_DIR/.env.client"
: "${API_BASE_URL:?set API_BASE_URL (or provide .env.client)}"; : "${API_KEY:?set API_KEY}"
S1="${STAGE1_TIMEOUT:-300}"; S2="${STAGE2_TIMEOUT:-120}"; MAXW="${MAX_WAIT:-1800}"
RES="$VAR/results"; mkdir -p "$RES"; : > "$RES/ALL_SUMMARIES.txt"
STAMP="$(date +%m%d%H%M)"

names=("$@"); [ ${#names[@]} -gt 0 ] || while IFS= read -r d; do names+=("$(basename "$d")"); done < <(find "$VAR" -maxdepth 1 -mindepth 1 -type d -name 'v*' | sort)

for v in "${names[@]}"; do
  id="diag-${v//_/-}-$STAMP"; out="$RES/$v"; mkdir -p "$out"
  echo "=== $v  (job $id)  stage1=${S1}s stage2=${S2}s"
  code=$(curl -sS -o "$out/post.json" -w '%{http_code}' -X POST "$API_BASE_URL/jobs" -H "X-API-Key: $API_KEY" \
      --form-string "job_id=$id" --form-string "scenario_name=$v" \
      --form-string stage1_solver=scip --form-string stage2_solver=scip \
      --form-string "stage1_timeout=$S1" --form-string "stage2_timeout=$S2" --form-string stage2_num_solutions_limit=1 \
      -F "environment=@$VAR/$v/EnvConfig.yaml;type=application/x-yaml" -F "schedule=@$VAR/$v/Schedule.yaml;type=application/x-yaml") || code=000
  if [ "$code" != "202" ]; then echo "  POST failed (HTTP $code): $(head -c 300 "$out/post.json" 2>/dev/null)" | tee -a "$RES/ALL_SUMMARIES.txt"; continue; fi
  t0=$(date +%s); st=""
  while :; do
    st=$(curl -sS -H "X-API-Key: $API_KEY" "$API_BASE_URL/jobs/$id/status" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["job_status"]["status"])' 2>/dev/null | tr -d '\r')
    case "$st" in finished|failed|canceled) break;; esac
    [ $(( $(date +%s) - t0 )) -gt "$MAXW" ] && { echo "  gave up waiting (>${MAXW}s) — cancelling"; curl -sS -X DELETE -H "X-API-Key: $API_KEY" "$API_BASE_URL/jobs/$id" >/dev/null; break; }
    sleep 10
  done
  echo "  -> $st after $(( $(date +%s) - t0 )) s"
  jd="$HOME_DIR/data/jobs/$id"
  if [ -d "$jd" ]; then
    for f in status.json result.json request.json logs/error.log logs/worker.log opt_output/optimization.log opt_output/result_summary.json opt_output/internal/scip_S1.log opt_output/internal/scip_S2.log; do
      [ -f "$jd/$f" ] && { mkdir -p "$out/$(dirname "$f")"; cp "$jd/$f" "$out/$f"; }
    done
    "$PY" "$HERE/summarize_run.py" "$out" > "$out/summary.txt" 2>&1
    cat "$out/summary.txt"; { echo "##### $v"; cat "$out/summary.txt"; echo; } >> "$RES/ALL_SUMMARIES.txt"
  else
    echo "  (no $jd — is <sigmai_compose_dir> the folder that holds data/?)" | tee -a "$RES/ALL_SUMMARIES.txt"
  fi
done
echo; echo "Done. Send: $RES/ALL_SUMMARIES.txt"
