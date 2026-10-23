#!/usr/bin/env bash
# A/B comparison (Task 19): for every input and repetition, the two builds run IN PARALLEL (two containers) under the same
# conditions. Not part of the product.
#
#   scripts/perf/ab.sh <tag> <build_A> <build_B> <reps> <input_dir> [<input_dir> ...]
#
# Needs PERF_RUNS and PERF_EXTRA (see run.sh). Run directories: <tag>-<inputname>-<A|B>-<rep>. Writes PERF_RUNS/<tag>.done.
set -u
TAG="${1:?tag}"; A="${2:?build A}"; B="${3:?build B}"; REPS="${4:?reps}"; shift 4
HERE="$(cd "$(dirname "$0")" && pwd)"
for IN in "$@"; do
  N="$(basename "${IN}")"
  for i in $(seq 1 "${REPS}"); do
    bash "${HERE}/run.sh" "${TAG}-${N}-A-${i}" "${A}" "${IN}" > /dev/null 2>&1 &
    bash "${HERE}/run.sh" "${TAG}-${N}-B-${i}" "${B}" "${IN}" > /dev/null 2>&1 &
    wait
  done
done
echo finished > "${PERF_RUNS}/${TAG}.done"
