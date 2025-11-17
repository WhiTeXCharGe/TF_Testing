from dataclasses import dataclass
import random
from datetime import date, timedelta
import yaml

from config_base import *

# ---------------- YAML helpers (inline list/dict) ----------------

class InlineList(list):
    pass


def represent_inline_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class InlineDict(dict):
    pass


def represent_inline_dict(dumper, data):
    # map tag (not seq), but still flow-style (inline)
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


def setup_yaml_inline():
    yaml.add_representer(InlineList, represent_inline_list)
    yaml.add_representer(InlineDict, represent_inline_dict)


# ---------------- Workflow / operations ----------------

@dataclass
class OperationDef:
    id: str
    name: str
    workhours: list[int]
    worker_range: tuple[int, int]

    def to_dict(self) -> dict:
        # Java parseEnv expects "work_hours"
        return {
            "id": self.id,
            "name": self.name,
            "work_hours": InlineList(self.workhours),
            "min_worker_num": self.worker_range[0],
            "max_worker_num": self.worker_range[1],
        }


@dataclass
class PhaseDef:
    id: str
    name: str
    operation_list: list[OperationDef]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "operation_list": [o.to_dict() for o in self.operation_list],
        }


@dataclass
class WorkflowDef:
    phase_list: list[PhaseDef]

    def to_dict(self) -> dict:
        return {
            "id": "workflow",
            "name": "Equipment Setup",
            "phase_list": [p.to_dict() for p in self.phase_list],
        }


def create_workflow() -> WorkflowDef:
    phase_names = [
        ("Module Setup", ["Heavy", "Mech", "Elec"]),
        ("Hardware Setup", ["Mech", "Elec"]),
        ("Function Setup", ["QC"]),
        ("Acceptance Inspection", ["QC", "Mech"]),
    ]

    phase_list: list[PhaseDef] = []
    for i, (phase_name, op_names) in enumerate(phase_names):
        phase_id = f"p{i + 1}"

        ope_list: list[OperationDef] = []
        for j, op_name in enumerate(op_names):
            ope_id = f"{phase_id}o{j + 1}"
            ope_list.append(
                OperationDef(
                    id=ope_id,
                    name=op_name,
                    workhours=[8, 10, 12],
                    worker_range=(2, 3),
                )
            )

        phase_list.append(PhaseDef(id=phase_id, name=phase_name, operation_list=ope_list))

    return WorkflowDef(phase_list)


# ---------------- Regions / companies / fabs ----------------

@dataclass
class RegionDef:
    id: str
    name: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "max_stay_on": 1000,
            "max_annual_stay": 1000,
            "stay_off_interval": 1000,
            "unavailable_dates": [],
        }


@dataclass
class CompanyDef:
    id: str
    name: str

    def to_customer_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "unavailable_dates": [],
        }

    def to_worker_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "annual_overtime_limit": 1000,
            "monthly_overtime_limit": 1000,
            "unavailable_dates": [],
        }


@dataclass
class FabDef:
    id: str
    name: str
    region: RegionDef
    company: CompanyDef

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "region": self.region.id,
            "customer_company": self.company.id,
            "unavailable_dates": [],
        }


def create_area_map() -> tuple[dict[str, FabDef], dict[str, RegionDef], dict[str, CompanyDef]]:
    region_map: dict[str, RegionDef] = {
        "r1": RegionDef("r1", "Japan"),
    }
    company_map: dict[str, CompanyDef] = {
        "c1": CompanyDef("c1", "AAA"),
    }

    fab_num = 3
    fab_map: dict[str, FabDef] = {}
    region = next(iter(region_map.values()))
    company = next(iter(company_map.values()))
    for i in range(fab_num):
        fab = FabDef(
            id=f"f{i + 1}",
            name=f"{company.name} Fab{i + 1}",
            region=region,
            company=company,
        )
        fab_map[fab.id] = fab

    return fab_map, region_map, company_map


def create_worker_company_map() -> dict[str, CompanyDef]:
    companies = [
        CompanyDef("c2", "XXX"),
        CompanyDef("c3", "YYY"),
    ]

    company_map: dict[str, CompanyDef] = {}
    # worker_company_num is from config_base
    for i in range(worker_company_num):
        company = companies[i]
        company_map[company.id] = company

    return company_map


# ---------------- Workers ----------------

@dataclass
class WorkerDef:
    id: str
    name: str
    company: CompanyDef
    is_manager: bool
    skill_map: list[tuple[OperationDef, int]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "worker_company": self.company.id,
            "is_manager": self.is_manager,
            "skill_map": InlineDict({op.id: lvl for (op, lvl) in self.skill_map}),
            "fab_suitability_map": [],
            "unavailable_dates": [],
        }


def create_worker_list(
    worker_num: int,
    worker_company_map: dict[str, CompanyDef],
    workflow: WorkflowDef,
) -> list[WorkerDef]:
    # All operations are possible skills
    skill_list: list[OperationDef] = [
        o for p in workflow.phase_list for o in p.operation_list
    ]
    skill_num_range = (3, 6)

    company_list = list(worker_company_map.values())

    worker_list: list[WorkerDef] = []
    for i in range(worker_num):
        wid = f"w{i + 1}"
        name = chr(ord("A") + i // 26) + chr(ord("A") + i % 26)

        # choose skill subset
        skill_num = random.randint(skill_num_range[0], skill_num_range[1])
        skill_index_list = sorted(random.sample(range(len(skill_list)), skill_num))
        skills = [skill_list[idx] for idx in skill_index_list]
        skill_levels = random.choices(
            skill_level_list,
            skill_level_weights,
            k=len(skills),
        )

        company = random.choice(company_list)
        # FIX: use random.choices for weighted manager flag
        is_manager = random.choices(
            [True, False],
            [manager_rate, 1.0 - manager_rate],
            k=1,
        )[0]

        worker = WorkerDef(
            id=wid,
            name=name,
            company=company,
            is_manager=is_manager,
            skill_map=list(zip(skills, skill_levels)),
        )
        worker_list.append(worker)

    return worker_list


# ---------------- Environment YAML writer ----------------

def write_environment_yaml(
    filepath: str,
    workflow: WorkflowDef,
    fab_map: dict[str, FabDef],
    region_map: dict[str, RegionDef],
    customer_company_map: dict[str, CompanyDef],
    worker_company_map: dict[str, CompanyDef],
    worker_list: list[WorkerDef],
):
    env_data: dict[str, object] = {}

    # workflow definitions
    env_data["workflow_list"] = [workflow.to_dict()]

    # fabs, regions, companies
    env_data["fab_list"] = [f.to_dict() for f in fab_map.values()]
    env_data["region_list"] = [r.to_dict() for r in region_map.values()]

    # FIX: key name "customer_company_list" (Java expects this)
    env_data["customer_company_list"] = [
        c.to_customer_dict() for c in customer_company_map.values()
    ]

    env_data["worker_company_list"] = [
        c.to_worker_dict() for c in worker_company_map.values()
    ]

    # FIX: key name "transite_day_map" (matches Java buildCalendars)
    env_data["transite_day_map"] = []

    # FIX: we need worker_list here, not overriding workflow_list
    env_data["worker_list"] = [w.to_dict() for w in worker_list]

    root_data = {
        "environment": env_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(root_data, f, allow_unicode=True, sort_keys=False)


# ---------------- Schedule generation ----------------

def is_holiday(day: date, _fab: FabDef | None) -> bool:
    if is_skip_weekend:
        # Saturday = 5, Sunday = 6
        if day.weekday() in (5, 6):
            return True
        # TODO: fab-specific holidays if needed
    return False


def to_eq_operation_dict(
    operation: OperationDef,
    workload_days: int,
    eq_id: str,
) -> dict:
    op_task_id = f"{eq_id}{operation.id}"  # e.g. e1p1o1
    return {
        "id": op_task_id,
        "name": operation.name,
        "operation": operation.id,
        "workload_days": workload_days,
    }


def to_eq_phase_dict(
    phase: PhaseDef,
    operation_worklengths: list[int],
    start_day: date,
    end_day: date,
    eq_id: str,
) -> dict:
    phase_task_id = f"{eq_id}{phase.id}"
    operation_task_list: list[dict] = []
    for op, wl in zip(phase.operation_list, operation_worklengths):
        operation_dict = to_eq_operation_dict(
            operation=op,
            workload_days=wl,
            eq_id=eq_id,
        )
        operation_task_list.append(operation_dict)

    return {
        "id": phase_task_id,
        "name": phase.name,
        "phase": phase.id,
        "start_date": start_day.strftime("%Y/%m/%d"),
        "end_date": end_day.strftime("%Y/%m/%d"),
        # FIX: key name "operation_task_list"
        "operation_task_list": operation_task_list,
    }


def to_eq_dict(
    index: int,
    workflow: WorkflowDef,
    fab: FabDef,
    start_day: date,
    worklength: list[tuple[int, list[int]]],
) -> tuple[dict, date]:
    # module id e1, e2, ...
    eq_id = f"e{index + 1}"
    # simple name: e.g. "SU 1001A"
    name = f"SU {1000 + index + 1}A"

    phase_task_list: list[dict] = []

    # We make phases sequential: p1 -> p2 -> p3 -> p4
    phase_start = start_day
    # ensure phase_start is not holiday
    while is_holiday(phase_start, fab):
        phase_start = phase_start + timedelta(days=1)

    final_end_day = phase_start

    for i, phase in enumerate(workflow.phase_list):
        phase_length_days = worklength[i][0]
        op_worklengths = worklength[i][1]

        # compute phase_end as phase_length_days *working days* after phase_start
        phase_end = phase_start
        days_needed = phase_length_days
        while days_needed > 0:
            phase_end = phase_end + timedelta(days=1)
            if not is_holiday(phase_end, fab):
                days_needed -= 1

        phase_dict = to_eq_phase_dict(
            phase=phase,
            operation_worklengths=op_worklengths,
            start_day=phase_start,
            end_day=phase_end,
            eq_id=eq_id,
        )
        phase_task_list.append(phase_dict)

        final_end_day = phase_end

        # next phase starts after this one
        phase_start = phase_end + timedelta(days=1)
        while is_holiday(phase_start, fab):
            phase_start = phase_start + timedelta(days=1)

    eq_dict = {
        "id": eq_id,
        "name": name,
        "workflow": "workflow",
        "fab": fab.id,
        "phase_task_list": phase_task_list,
    }

    return eq_dict, final_end_day


def create_worklength_list(
    _workflow: WorkflowDef,
) -> tuple[list[list[tuple[int, list[int]]]], list[float]]:
    # Each entry in normal_worklength/vip_worklength:
    #   (phase_total_days, [workload_days_per_operation])
    # They must have length = number of phases.
    return (
        [normal_worklength, vip_worklength],
        [0.8, 0.2],
    )


def create_equipment_list(
    workflow: WorkflowDef,
    fab_map: dict[str, FabDef],
    eq_per_day: float,
    eq_per_day_sigma: float,
    eq_num: int,
    start_day: date,
) -> tuple[list[dict], date]:
    (worklength_list, worklength_weights) = create_worklength_list(workflow)

    fab_list = list(fab_map.values())

    eq_count = 0
    eq_point = max(1.0, eq_per_day)
    current_day = start_day
    end_day = start_day

    equipment_list: list[dict] = []

    while True:
        # spawn as many modules as eq_point allows today
        while eq_point >= 1.0 and eq_count < eq_num:
            worklength = random.choices(worklength_list, worklength_weights, k=1)[0]
            fab = random.choice(fab_list)

            eq_start_day = current_day
            while is_holiday(eq_start_day, fab):
                eq_start_day = eq_start_day + timedelta(days=1)

            eq, eq_end_day = to_eq_dict(
                index=eq_count,
                workflow=workflow,
                fab=fab,
                start_day=eq_start_day,
                worklength=worklength,
            )
            equipment_list.append(eq)

            eq_count += 1
            eq_point -= 1.0

            if end_day < eq_end_day:
                end_day = eq_end_day

            if eq_count >= eq_num:
                break

        if eq_count >= eq_num:
            break

        # move to next non-holiday day
        while True:
            current_day = current_day + timedelta(days=1)
            if not is_holiday(current_day, None):
                break

        # add new eq_point for the next day (Poisson-ish)
        point_diff = max(0.0, random.gauss(eq_per_day, eq_per_day_sigma))
        eq_point += point_diff

    return equipment_list, end_day


def write_schedule_yaml(
    filepath: str,
    equipment_list: list[dict],
    start_day: date,
    end_day: date,
):
    schedule_data: dict[str, object] = {}
    schedule_data["plan_range"] = {
        "start_date": start_day.strftime("%Y/%m/%d"),
        "end_date": end_day.strftime("%Y/%m/%d"),
    }

    schedule_data["workflow_task_list"] = equipment_list

    # FIX: correct key "assignment_list"
    schedule_data["assignment_list"] = []

    root_data = {
        "schedule": schedule_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(root_data, f, allow_unicode=True, sort_keys=False)


# ---------------- main ----------------

def main():
    ENV_SEED = 0
    EQ_SEED = 1

    START_DAY = date(2025, 9, 1)

    setup_yaml_inline()

    # ----- Env / workers -----
    random.seed(ENV_SEED)

    workflow = create_workflow()
    fab_map, region_map, customer_company_map = create_area_map()
    worker_company_map = create_worker_company_map()

    worker_list = create_worker_list(WORKER_NUM, worker_company_map, workflow)

    write_environment_yaml(
        "EnvConfig.yaml",
        workflow,
        fab_map,
        region_map,
        customer_company_map,
        worker_company_map,
        worker_list,
    )

    # ----- Schedule / equipment -----
    random.seed(EQ_SEED)

    equipment_list, end_day = create_equipment_list(
        workflow,
        fab_map,
        EQ_PER_DAYS,
        EQ_PER_DAYS_SIGMA,
        EQ_NUM,
        START_DAY,
    )

    # add +3 days buffer to plan_range.end_date
    write_schedule_yaml(
        "Schedule.yaml",
        equipment_list,
        START_DAY,
        end_day + timedelta(days=3),
    )


if __name__ == "__main__":
    main()
