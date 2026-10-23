#!/usr/bin/env python3
"""One line per run directory: local search time, steps, steps/s, move evaluation speed, best score (Task 19).

usage: summary.py <run dir> [<run dir> ...]
Reads console.txt of scripts/perf/run.sh (needs the Timefold INFO phase lines).
"""
import os
import re
import sys

PAT = re.compile(r"Local Search phase \(\d+\) ended: time spent \((\d+)\).*best score \(([^)]*)\), "
                 r"move evaluation speed \((\d+)/sec\), step total \((\d+)\)")
CH = re.compile(r"Construction Heuristic phase \((\d+)\) ended: time spent \((\d+)\)")

print("%-22s %8s %8s %8s %10s  %s" % ("run", "LS ms", "steps", "steps/s", "evals/s", "best score"))
for d in sys.argv[1:]:
    p = os.path.join(d, "console.txt")
    txt = open(p, encoding="utf-8", errors="replace").read()
    m = PAT.search(txt)
    name = os.path.basename(os.path.normpath(d))
    if not m:
        print("%-22s (no local search line)" % name)
        continue
    ms, score, evals, steps = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    print("%-22s %8d %8d %8.1f %10d  %s" % (name, ms, steps, steps * 1000.0 / ms, evals, score))

# best score over time, when the run kept score_progress.txt (write_score_files=true)
for d in sys.argv[1:]:
    p = os.path.join(d, "score_progress.txt")
    if not os.path.exists(p):
        continue
    rows = []
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"Stage1\s+\| t\+\s*(\d+)s \| total=(\S+)", line)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    print("\nscore_progress of %s (%d improvements)" % (os.path.basename(os.path.normpath(d)), len(rows)))
    for t in (10, 30, 60, 90, 120, 150, 180):
        best = [s for (ts, s) in rows if ts <= t]
        if best:
            print("  t<=%3ds  %s" % (t, best[-1]))
