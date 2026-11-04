#!/usr/bin/env python3
"""Summarize one Sigma-I job folder (data/jobs/<job_id>/) in ~30 lines.

usage: python summarize_run.py <job_dir> [--json]

Reads status.json, result.json, opt_output/optimization.log, opt_output/internal/scip_S1.log
(and scip_S2.log if present). Never loads result_context.pkl.
"""
import json, re, sys
from pathlib import Path

def read(p):
    try: return Path(p).read_text(encoding="utf-8", errors="replace")
    except OSError: return ""

def jload(p):
    try: return json.loads(read(p))
    except Exception: return {}

def scip_summary(text):
    out = {}
    for key, pat in (("scip_status", r"SCIP Status\s*:\s*(.+)"), ("solve_sec", r"Solving Time \(sec\)\s*:\s*([\d.]+)"),
                     ("primal", r"Primal Bound\s*:\s*([+\-\d.e]+)"), ("dual", r"Dual Bound\s*:\s*([+\-\d.e]+)"),
                     ("gap", r"Gap\s*:\s*([\d.]+) %")):
        m = list(re.finditer(pat, text)); out[key] = m[-1].group(1).strip() if m else None
    # when did the table first show the final primal value?
    rows = re.findall(r"^\s*(\d+)s\|[^\n]*?\|\s*([-+\d.e]+)\s*\|\s*([-+\d.e]+)\s*\|\s*[\d.]+%", text, re.M)
    out["first_sec_at_final_primal"] = None
    if rows and out["primal"]:
        try:
            final = float(out["primal"])
            for sec, _dual, primal in rows:
                if abs(float(primal) - final) <= abs(final) * 1e-6:
                    out["first_sec_at_final_primal"] = int(sec); break
        except ValueError: pass
    return out

def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    d = Path(sys.argv[1]); as_json = "--json" in sys.argv
    st, rs = jload(d / "status.json"), jload(d / "result.json")
    log = read(d / "opt_output/optimization.log")
    res = {
        "job_id": st.get("job_id") or d.name, "status": st.get("status"), "api_error": (st.get("error") or {}).get("type"),
        "core_status": (rs.get("meta") or {}).get("core_status"), "validation_passed": (rs.get("meta") or {}).get("validation_passed"),
        "stage1_status": (rs.get("result") or {}).get("stage1_result", {}).get("status"),
        "stage2_status": (rs.get("result") or {}).get("stage2_result", {}).get("status"),
        "stage1_sec": (rs.get("summary") or {}).get("stage1_solve_time_sec"),
        "stage2_sec": (rs.get("summary") or {}).get("stage2_solve_time_sec"),
    }
    m = re.search(r"Data Prep Completed: (\d+) tasks, (\d+) workers", log)
    res["tasks"], res["workers"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    res["no_Y_phases (S1-C1)"] = sorted(set(re.findall(r"S1-C1: phase (\S+) に有効な Y 変数がありません", log)))
    ig = re.findall(r"\[Integrity/S1-C1\] 未割り当てのphaseがあります \(計(\d+)件\)", log)
    res["integrity_unassigned_count"] = int(ig[-1]) if ig else 0
    m = re.search(r"Existing plan input diagnostics: (\d+) issue", log); res["existing_plan_issues"] = int(m.group(1)) if m else 0
    res["existing_plan_issue_reasons"] = dict(sorted({r: log.count(f"'reason': '{r}'") for r in set(re.findall(r"'reason': '([^']+)'", log))}.items()))
    for s in ("S1", "S2"):
        t = read(d / f"opt_output/internal/scip_{s}.log")
        if t: res[f"scip_{s}"] = scip_summary(t)
    m = re.findall(r"総変数: ([\d,]+), 総制約数: ([\d,]+)", log); res["model_vars_cons"] = m[0] if m else None
    if as_json: print(json.dumps(res, ensure_ascii=False, indent=2)); return
    print(f"== {res['job_id']}  status={res['status']}  api_error={res['api_error']}  core_status={res['core_status']}")
    print(f"   stage1={res['stage1_status']} ({res['stage1_sec']} s)   stage2={res['stage2_status']!r} ({res['stage2_sec']} s)   validation_passed={res['validation_passed']}")
    print(f"   data: {res['tasks']} tasks, {res['workers']} workers; S1 model vars/cons: {res['model_vars_cons']}")
    print(f"   phases with NO valid start variable (S1-C1) : {len(res['no_Y_phases (S1-C1)'])} -> {', '.join(res['no_Y_phases (S1-C1)']) or '-'}")
    print(f"   integrity check: unassigned phases after solve = {res['integrity_unassigned_count']}")
    print(f"   existing-plan diagnostics: {res['existing_plan_issues']} (reasons seen in the log excerpt only: {res['existing_plan_issue_reasons']})")
    for s in ("S1", "S2"):
        k = res.get(f"scip_{s}")
        if k: print(f"   SCIP {s}: {k['scip_status']} | time {k['solve_sec']} s | primal {k['primal']} dual {k['dual']} gap {k['gap']}% | final primal first seen at {k['first_sec_at_final_primal']} s")

main()
