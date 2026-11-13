"""A small EnvConfig-shaped dict for the tests of tool 2-4 (same structure as a real `environment`, neutral names)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # the tool folder (env_*.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # weekly_planning/ (the folder of common)

from common.seiban_list import Template, TemplateOp   # noqa: E402


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


__all__ = ["make_env", "template", "TEMPLATE_LIKE_WF_TOOL", "TEMPLATE_WITH_P1"]
