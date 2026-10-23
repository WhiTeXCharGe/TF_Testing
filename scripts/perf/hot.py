#!/usr/bin/env python3
"""Aggregate `jfr print --events jdk.ExecutionSample --stack-depth 64` output (Task 19).

usage: hot.py samples.txt [top_n]
Prints: top frames (self time), inclusive time of project methods/classes (com.yourorg), inclusive time of
constraint classes. Percentages are of all samples of the main (solver) thread.
"""
import re
import sys
from collections import Counter

path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

samples = []
cur = None
in_stack = False
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        s = line.strip()
        if s.startswith("jdk.ExecutionSample"):
            cur = []
            continue
        if cur is None:
            continue
        if s.startswith("stackTrace = ["):
            in_stack = True
            continue
        if in_stack:
            if s == "]":
                in_stack = False
                samples.append(cur)
                cur = None
            elif s and not s.startswith("..."):
                cur.append(re.sub(r"\s+line:.*$", "", s))

n = len(samples)
self_c = Counter(st[0] for st in samples if st)
incl = Counter()
incl_cls = Counter()
for st in samples:
    seen = set(st)
    for m in seen:
        if m.startswith("com.yourorg"):
            incl[m] += 1
    seen_cls = {m.rsplit(".", 1)[0] for m in st if m.startswith("com.yourorg")}
    for c in seen_cls:
        incl_cls[c] += 1

nearest = Counter()
for st in samples:
    for m in st:
        if m.startswith("com.yourorg"):
            nearest[m] += 1
            break

def pct(x):
    return "%5.1f%%" % (100.0 * x / n)

print("samples: %d" % n)
print("\n== top %d frames (self) ==" % top_n)
for m, c in self_c.most_common(top_n):
    print(pct(c), m)
print("\n== project classes, inclusive ==")
for m, c in incl_cls.most_common(top_n):
    print(pct(c), m)
print("\n== project methods, inclusive ==")
for m, c in incl.most_common(top_n * 2):
    print(pct(c), m)
print("\n== nearest project frame of each sample (own code plus what it calls) ==")
for m, c in nearest.most_common(top_n * 2):
    print(pct(c), m)

# optional: callers of the hottest self frame, and inclusive share of extra substrings (argv[3:])
print("\n== callers (first non-JDK frame above the top frame) of the top self frame ==")
top_frame = self_c.most_common(1)[0][0] if self_c else None
callers = Counter()
for st in samples:
    if st and st[0] == top_frame:
        for m in st[1:]:
            if not m.startswith(("java.", "jdk.", "sun.")):
                callers[m] += 1
                break
for m, c in callers.most_common(8):
    print(pct(c), m)
extra = sys.argv[3:]
if extra:
    print("\n== inclusive share of the given substrings ==")
    for sub in extra:
        c = sum(1 for st in samples if any(sub in m for m in st))
        print(pct(c), sub)
