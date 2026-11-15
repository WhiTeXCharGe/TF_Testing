"""A small EnvConfig-shaped dict for the tests of tool 2-4 (same structure as a real `environment`, neutral names)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # the tool folder (env_*.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # weekly_planning/ (the folder of common)

from common import seiban_list as sl   # noqa: E402
from common.seiban_list import PlanRow, SeibanList, Template, TemplateOp   # noqa: E402


def region(rid, name):
    return {"id": rid, "name": name, "max_stay_on": 10000, "max_annual_stay": 10000, "stay_off_interval": 3, "unavailable_dates": []}


def op(op_id, name, lo=1, hi=8):
    return {"id": op_id, "name": name, "work_hours": [10], "min_worker_num": lo, "max_worker_num": hi}


def make_env():
    regions = [region("r_other", "Other"), region("r1", "Region X"), region("r2", "Region Y")]
    pairs = [{"from": a["id"], "to": b["id"], "days": 1} for a in regions for b in regions if a is not b]
    return {
        "workflow_list": [{"id": "wf_tool", "name": "Tool Install", "phase_list": [
            {"id": "p2", "name": "Hardware Setup", "operation_list": [op("p2o1", "Mech"), op("p2o2", "Elec")]},
            {"id": "p3", "name": "Function Setup", "operation_list": [op("p3o1", "QC")]},
            {"id": "p4", "name": "Acceptance Inspection", "operation_list": [op("p4o1", "QC")]}]}],
        "fab_list": [
            {"id": "f_other", "name": "Other", "region": "r_other", "customer_company": "c_other", "unavailable_dates": []},
            {"id": "f1", "name": "Fab A", "region": "r1", "customer_company": "c1", "unavailable_dates": []},
            {"id": "f2", "name": "Fab B", "region": "r2", "customer_company": "c2", "unavailable_dates": []}],
        "region_list": regions,
        "customer_company_list": [{"id": "c_other", "name": "OTHER", "unavailable_dates": []},
                                  {"id": "c1", "name": "Cust A", "unavailable_dates": []},
                                  {"id": "c2", "name": "Cust B", "unavailable_dates": []}],
        "worker_company_list": [{"id": "wc1", "name": "Company 1"}],
        "transite_day_map": pairs,
        "worker_list": [
            {"id": "w001", "name": "Worker 1", "worker_company": "wc1", "is_manager": False,
             "skill_map": {"p2o1": 2, "p2o2": 0, "p3o1": 3, "p4o1": 1, "other_op": 1, "personal_business_op": 1},
             "worker_type_by_operation": {"p2o1": "regular", "p3o1": "spot", "p4o1": "regular"}, "fab_suitability_map": []},
            {"id": "w002", "name": "Worker 2", "worker_company": "wc1", "is_manager": False,
             "skill_map": {"p2o1": 1, "p2o2": 1, "p3o1": 0, "p4o1": 0, "other_op": 1, "personal_business_op": 1},
             "worker_type_by_operation": {"p2o1": "spot"}, "fab_suitability_map": []},
            {"id": "w003", "name": "Worker 3", "worker_company": "wc1", "is_manager": False,
             "skill_map": {"other_op": 1, "personal_business_op": 1}, "worker_type_by_operation": {}, "fab_suitability_map": []}],
    }


def template(number, ops):
    """ops: [(phase, op, name, hours, min, max)]"""
    return Template(number, f"工程テンプレート{number}", [TemplateOp(*o) for o in ops])


# the same operations as wf_tool (min 1 / max 8), as a template
TEMPLATE_LIKE_WF_TOOL = [(2, 1, "Mech", 576, 1, 8), (2, 2, "Elec", 144, 1, 8), (3, 1, "QC", 64, 1, 8), (4, 1, "QC", 288, 1, 8)]
# a template with a process 1 and other minimum / maximum workers
TEMPLATE_WITH_P1 = [(1, 1, "Mech", 160, 2, 4), (1, 2, "Elec", 40, 1, 1), (2, 1, "Mech", 576, 2, 4), (4, 1, "QC", 288, 1, 2)]


def _task(module, code, fab, phases):
    """A workflow task of workflow wf_tool; phases = {P: [(K, hours)]}."""
    names = {2: "Hardware Setup", 3: "Function Setup", 4: "Acceptance Inspection"}
    ops = {(2, 1): "Mech", (2, 2): "Elec", (3, 1): "QC", (4, 1): "QC"}
    return {"id": module, "name": code, "description": None, "workflow": "wf_tool", "fab": fab, "phase_task_list": [
        {"id": f"{module}p{p}", "name": names[p], "description": None, "phase": f"p{p}", "start_date": "2026/12/01", "end_date": "2026/12/10",
         "operation_task_list": [{"id": f"{module}p{p}o{k}", "name": ops[(p, k)], "description": None, "operation": f"p{p}o{k}",
                                  "workload_hours": h, "recommends_worker_min": 1, "recommends_worker_max": 2, "color_code": None}
                                 for k, h in items]} for p, items in phases.items()]}


def _assignment(worker, op_task, flexibility="Flexible", day="2026/12/02"):
    return {"worker": worker, "operation_task": op_task, "start_date": day, "end_date": day,
            "work_date_list": [{"date": day, "hour": 10}], "plan_flexibility": flexibility, "description": None}


def make_schedule():
    """Two flexible modules (e1 = 530N00001A, e2 = 530N00002A), a misc task and their assignments.  e1 has a Fixed row."""
    phases = {2: [(1, 576), (2, 144)], 3: [(1, 64)], 4: [(1, 288)]}
    return {"schedule": {
        "plan_range": {"start_date": "2026/12/01", "end_date": "2027/03/31"},
        "workflow_task_list": [_task("e1", "530N00001A", "f1", phases), _task("e2", "530N00002A", "f2", phases)],
        "misc_task_list": [{"id": "misc_other_1", "name": "Other", "description": None, "workflow": None, "region": "r1"}],
        "assignment_list": [_assignment("w001", "e1p2o1"), _assignment("w002", "e1p2o1", "Fixed"), _assignment("w001", "e1p3o1"),
                            _assignment("w001", "e2p2o1"), _assignment("w002", "misc_other_1")]}}


DATES = {1: ("2026/12/01", "2026/12/10"), 2: ("2026/12/05", "2026/12/20"), 3: ("2026/12/15", "2026/12/25"), 4: ("2026/12/22", "2027/01/10")}


def plan_row(excel_row=2, kind=sl.ADD, code="530N00005A", customer="Cust A", fab="Fab A", region="Region X", template=1, phases=None, problems=None):
    """A row of 計画対象 as the reader gives it (a white row of template 1 by default)."""
    return PlanRow(excel_row, kind, "none" if kind == sl.ADD else "FFDCE6", code, code or "", customer, fab, region, template,
                   f"工程テンプレート{template}", dict(DATES if phases is None else phases), list(problems or []))


def seiban_data(rows, templates=None, warnings=None):
    """A SeibanList; template 1 = the operations of wf_tool, template 2 = TEMPLATE_WITH_P1 unless `templates` is given."""
    templates = templates or {1: template(1, TEMPLATE_LIKE_WF_TOOL), 2: template(2, TEMPLATE_WITH_P1)}
    return SeibanList(rows, [], templates, list(warnings or []))


__all__ = ["make_env", "make_schedule", "template", "plan_row", "seiban_data", "DATES", "TEMPLATE_LIKE_WF_TOOL", "TEMPLATE_WITH_P1"]
