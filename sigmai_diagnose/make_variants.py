#!/usr/bin/env python3
"""Make test variants of EnvConfig.yaml / Schedule.yaml to find what triggers S1_INVALID.

usage:
  python make_variants.py <input_dir> <out_dir> [--bad-phases e1p2,e9p2,...]

<input_dir> holds the original EnvConfig.yaml + Schedule.yaml. For each variant a folder
<out_dir>/<name>/ is written with EnvConfig.yaml (always unchanged) and Schedule.yaml (edited),
plus <out_dir>/VARIANTS.md describing what each variant tests.

Only the assignment_list of Schedule.yaml is edited; nothing else is touched. Each output is
re-loaded and compared with the intended content before it is written.
"""
import argparse, copy, shutil, sys
from pathlib import Path
import yaml

DEFAULT_BLOCKED = "e1p2,e9p2,e9p3,e10p3,e20p3,e26p4"   # phases where the Fixed worker provably cannot extend the run (see guide 06)
DEFAULT_BAD = "e1p2,e9p2,e9p3,e10p3,e11p2,e11p3,e14p3,e16p2,e19p2,e20p2,e20p3,e20p4,e24p2,e24p3,e26p4"

def dump(doc, path):
    text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=1_000_000, default_flow_style=None)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    again = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if again != doc:
        sys.exit(f"ERROR: re-loading {path} gives different content — refusing to continue")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("input_dir"); ap.add_argument("out_dir")
    ap.add_argument("--bad-phases", default=DEFAULT_BAD)
    ap.add_argument("--blocked-phases", default=DEFAULT_BLOCKED)
    a = ap.parse_args()
    src, out = Path(a.input_dir), Path(a.out_dir)
    bad = {p for p in a.bad_phases.split(",") if p}
    blocked = {p for p in a.blocked_phases.split(",") if p}
    doc = yaml.safe_load((src / "Schedule.yaml").read_text(encoding="utf-8"))
    sch = doc["schedule"]
    op2ph = {o["id"]: p["id"] for wf in sch["workflow_task_list"] for p in wf["phase_task_list"] for o in p["operation_task_list"]}
    al = sch["assignment_list"]
    is_phase = lambda x: x["operation_task"] in op2ph
    n_phase = sum(is_phase(x) for x in al)

    def variant(name, desc, fn):
        d2 = copy.deepcopy(doc); s2 = d2["schedule"]; s2["assignment_list"] = fn(s2["assignment_list"])
        vd = out / name; vd.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "EnvConfig.yaml", vd / "EnvConfig.yaml")
        dump(d2, vd / "Schedule.yaml")
        return (name, desc, len(al), len(s2["assignment_list"]))

    def fixed_to_flex(lst):
        r = copy.deepcopy(lst)
        for x in r:
            if is_phase(x) and x["plan_flexibility"] == "Fixed": x["plan_flexibility"] = "Flexible"
        return r

    def fixed_to_flex_blocked(lst):
        r = copy.deepcopy(lst)
        for x in r:
            if op2ph.get(x["operation_task"]) in blocked and x["plan_flexibility"] == "Fixed": x["plan_flexibility"] = "Flexible"
        return r

    rows = [
        variant("v0_baseline", "Unchanged. Control: must reproduce the same 'no valid Y' phases as the failed run.", lambda l: copy.deepcopy(l)),
        variant("v1_phase_fixed_to_flexible", f"Only the Fixed rows that belong to workflow phases (not misc) become Flexible.", fixed_to_flex),
        variant("v2_no_phase_assignments", f"Delete all {n_phase} assignment rows of workflow phases; keep the misc (other-work) rows.", lambda l: [x for x in l if not is_phase(x)]),
        variant("v3_no_misc_assignments", "Delete the misc (other-work) rows; keep the phase rows.", lambda l: [x for x in l if is_phase(x)]),
        variant("v4_drop_failing_phase_rows", f"Delete only the assignment rows of the {len(bad)} failing phases.", lambda l: [x for x in l if op2ph.get(x["operation_task"]) not in bad]),
        variant("v6_flexible_only_blocked_phases", f"Fixed -> Flexible only in the {len(blocked)} phases proven blocked ({', '.join(sorted(blocked))}); the other failing phases stay as they are.", fixed_to_flex_blocked),
        variant("v5_no_assignments_at_all", "Delete every assignment row (plan from scratch). Sanity check: should solve.", lambda l: []),
    ]
    md = ["# Variants", "", f"Source: `{src}` - failing phases assumed: {', '.join(sorted(bad))}", "",
          "| variant | rows in -> out | what it tests |", "|---|---|---|"]
    md += [f"| `{n}` | {i} -> {o} | {d} |" for n, d, i, o in rows]
    (out / "VARIANTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

main()
