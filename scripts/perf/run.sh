#!/usr/bin/env bash
# Fixed-budget stage 1 measurement run (Task 19). Not part of the product.
#
#   scripts/perf/run.sh <label> <build_dir> <input_dir> [jfr=0|1] [score_files=false|true]
#
#   build_dir : a copy of target/ after "mvn -DskipTests package" (needs classes/ and dependency/)
#   input_dir : a directory with EnvConfig.yaml and Schedule.yaml (read only, copied into the run directory)
# Environment:
#   PERF_RUNS  scratch directory for the run directories (required)
#   PERF_EXTRA directory with slf4j-simple-2.0.18.jar (required). It is only on the measurement classpath, NOT in the pom:
#              it makes Timefold print its INFO phase lines ("move evaluation speed", "step total").
#
# Budget (the Task 19 ruling): stage 1 = 2 min, unimproved limit 600 s, stage 2 off, seed 42, stop_when_feasible=false,
# so the time limit always decides. Container: 4 CPUs, 7 GB, -Xmx5g, named perf-<label> (kept, then removed here).
# Results: PERF_RUNS/<label>/console.txt (+ rec.jfr, + score files when requested).
set -euo pipefail
LABEL="${1:?label}"; BUILD="${2:?build dir}"; INPUT="${3:?input dir}"; JFR="${4:-0}"; SCORE_FILES="${5:-false}"
PERF_RUNS="${PERF_RUNS:?set PERF_RUNS to a scratch dir}"
PERF_EXTRA="${PERF_EXTRA:?set PERF_EXTRA to the dir with slf4j-simple}"
OUT="${PERF_RUNS}/${LABEL}"
rm -rf "${OUT}"; mkdir -p "${OUT}/in"
cp "${INPUT}/EnvConfig.yaml" "${INPUT}/Schedule.yaml" "${OUT}/in/"
cat > "${OUT}/in/SolverParams.yaml" <<EOF
stage1_minutes: 2
stage1_unimproved_seconds: 600
stage1_stop_when_feasible: false
stage2_enabled: false
random_seed: 42
write_score_files: ${SCORE_FILES}
EOF
JFR_OPT=""
if [ "${JFR}" = "1" ]; then
  JFR_OPT="-XX:StartFlightRecording=filename=/run/rec.jfr,settings=profile,dumponexit=true"
fi
export MSYS_NO_PATHCONV=1
W_BUILD="$(cd "${BUILD}" && (pwd -W 2>/dev/null || pwd))"
W_OUT="$(cd "${OUT}" && (pwd -W 2>/dev/null || pwd))"
W_EXTRA="$(cd "${PERF_EXTRA}" && (pwd -W 2>/dev/null || pwd))"
NAME="perf-${LABEL}"
docker rm -f "${NAME}" > /dev/null 2>&1 || true
docker run --name "${NAME}" --cpus=4 --memory=7g \
  -v "${W_BUILD}:/app:ro" -v "${W_EXTRA}:/extra:ro" -v "${W_OUT}:/run" -w /run \
  maven:3.9-eclipse-temurin-21 \
  java -Xmx5g -Xms2g ${JFR_OPT} \
    -Dorg.slf4j.simpleLogger.defaultLogLevel=warn \
    -Dorg.slf4j.simpleLogger.log.ai.timefold.solver=info \
    -cp "/app/classes:/app/dependency/*:/extra/*" \
    com.yourorg.scheduler.Main in/EnvConfig.yaml in/Schedule.yaml --params-file=in/SolverParams.yaml \
  > "${OUT}/console.txt" 2>&1 || true
docker rm -f "${NAME}" > /dev/null 2>&1 || true
echo "== ${LABEL} =="
grep -E "Local Search phase .* ended|Stage1 done" "${OUT}/console.txt" | sed -e 's/^/  /' | cut -c1-300
