#!/usr/bin/env python3
"""A/B table of the runs of scripts/perf/ab.sh (Task 19).

usage: compare.py <runs dir> <tag> [<tag> ...]

For every run directory <tag>-<input>-<A|B>-<rep> it reads the "Local Search phase ... ended" line of console.txt and prints
steps/s, move evaluations/s and the final score per input and repetition, and the ratio B/A of each pair. The two runs of
a pair ran at the same time on the same machine, so the ratio is the figure to read; absolute speeds differ from round to
round (JIT, other containers, trajectory), up to about 15 percent.
"""
import os
import re
import sys

PAT = re.compile(r"Local Search phase \(\d+\) ended: time spent \((\d+)\).*best score \(([^)]*)\), "
                 r"move evaluation speed \((\d+)/sec\), step total \((\d+)\)")
NAME = re.compile(r"^(?P<tag>.+)-(?P<inp>[a-z_]+)-(?P<ab>[AB])-(?P<rep>\d+)$")


def read(d):
    try:
        txt = open(os.path.join(d, "console.txt"), encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = PAT.search(txt)
    if not m:
        return None
    ms, score, evals, steps = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    return {"steps_s": steps * 1000.0 / ms, "evals_s": float(evals), "score": score}


def main():
    root = sys.argv[1]
    for tag in sys.argv[2:]:
        rows = {}
        for name in sorted(os.listdir(root)):
            m = NAME.match(name)
            if not m or m.group("tag") != tag:
                continue
            r = read(os.path.join(root, name))
            if r:
                rows[(m.group("inp"), int(m.group("rep")), m.group("ab"))] = r
        print("== %s ==" % tag)
        print("%-16s %3s | %8s %8s %6s | %8s %8s %6s | %s" %
              ("input", "rep", "A st/s", "B st/s", "B/A", "A ev/s", "B ev/s", "B/A", "final score A  |  B"))
        ratios = {}
        for (inp, rep, ab) in sorted(rows):
            if ab != "A" or (inp, rep, "B") not in rows:
                continue
            a, b = rows[(inp, rep, "A")], rows[(inp, rep, "B")]
            rs, re_ = b["steps_s"] / a["steps_s"], b["evals_s"] / a["evals_s"]
            ratios.setdefault(inp, []).append((rs, re_))
            print("%-16s %3d | %8.1f %8.1f %6.3f | %8.0f %8.0f %6.3f | %s  |  %s" %
                  (inp, rep, a["steps_s"], b["steps_s"], rs, a["evals_s"], b["evals_s"], re_, a["score"], b["score"]))
        for inp, rr in sorted(ratios.items()):
            print("  %-16s B/A steps/s %s  evals/s %s" % (
                inp, " ".join("%.3f" % x[0] for x in rr), " ".join("%.3f" % x[1] for x in rr)))


if __name__ == "__main__":
    main()
