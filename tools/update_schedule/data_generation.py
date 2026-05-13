from dataclasses import dataclass, field
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
                    worker_range=(operation_worker_min, operation_worker_max),
                )
            )

        phase_list.append(PhaseDef(id=phase_id, name=phase_name, operation_list=ope_list))

    return WorkflowDef(phase_list)


# ---------------- Regions / companies / fabs ----------------

@dataclass
class RegionDef:
    id: str
    name: str
    max_stay_on: int
    max_annual_stay: int
    stay_off_interval: int
    weekly_weekdays: list[str]
    single_days: list[str]

    def to_dict(self) -> dict:
        unavailable: list[dict] = []

        if self.weekly_weekdays:
            unavailable.append({
                "weekly": {
                    "weekdays": InlineList(self.weekly_weekdays),
                }
            })

        if self.single_days:
            unavailable.append({
                "single": {
                    "days": self.single_days,
                }
            })

        return {
            "id": self.id,
            "name": self.name,
            "max_stay_on": self.max_stay_on,
            "max_annual_stay": self.max_annual_stay,
            "stay_off_interval": self.stay_off_interval,
            "unavailable_dates": unavailable,
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
            "annual_overtime_limit": 360,
            "monthly_overtime_limit": 40,
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
    # Build regions from config; id is assigned by list order
    region_map: dict[str, RegionDef] = {}
    for i, rdef in enumerate(region_definitions):
        rid = f"r{i + 1}"
        region_map[rid] = RegionDef(
            id=rid,
            name=rdef["name"],
            max_stay_on=rdef["max_stay_on"],
            max_annual_stay=rdef["max_annual_stay"],
            stay_off_interval=rdef["stay_off_interval"],
            weekly_weekdays=rdef["weekly_weekdays"],
            single_days=rdef["single_days"],
        )

    company_map: dict[str, CompanyDef] = {
        "c1": CompanyDef("c1", "AAA"),
        "c2": CompanyDef("c2", "BBB"),
    }

    fab_map: dict[str, FabDef] = {}

    # all region × all customer_company combinations
    fab_index = 1
    for region in region_map.values():
        for company in company_map.values():
            fab_id = f"f{fab_index}"
            fab_name = f"{company.name} Fab{fab_index}"
            fab = FabDef(
                id=fab_id,
                name=fab_name,
                region=region,
                company=company,
            )
            fab_map[fab.id] = fab
            fab_index += 1

    return fab_map, region_map, company_map


def create_worker_company_map() -> dict[str, CompanyDef]:
    companies = [
        CompanyDef("c3", "XXX"),
        CompanyDef("c4", "YYY"),
    ]

    company_map: dict[str, CompanyDef] = {}
    for i in range(worker_company_num):
        company = companies[i]
        company_map[company.id] = company

    return company_map


def create_transit_day_map(region_map: dict[str, RegionDef]) -> list[dict]:
    """Generate transit entries for all ordered region pairs using configured options."""
    transit_map = []
    region_ids = list(region_map.keys())
    for from_id in region_ids:
        for to_id in region_ids:
            if from_id != to_id:
                days = random.choices(transit_day_options, transit_day_weights, k=1)[0]
                transit_map.append({"from": from_id, "to": to_id, "days": days})
    return transit_map


# ---------------- Workers ----------------

@dataclass
class WorkerDef:
    id: str
    name: str
    company: CompanyDef
    is_manager: bool
    role: str
    skill_map: list[tuple[OperationDef, int]]
    worker_type_by_operation: dict[str, str]
    region_suitability: dict[str, int]
    customer_suitability: dict[str, int]
    unavailable_dates: list[str]
    affinity: list[str]

    def to_dict(self) -> dict:
        # Wrap all dates under a single "single" entry; empty list stays []
        unavailable: list[dict] = (
            [{"single": {"days": self.unavailable_dates}}]
            if self.unavailable_dates
            else []
        )

        return {
            "id": self.id,
            "name": self.name,
            "worker_company": self.company.id,
            "is_manager": self.is_manager,
            "role": self.role,
            "skill_map": InlineDict({op.id: lvl for (op, lvl) in self.skill_map}),
            "worker_type_by_operation": dict(self.worker_type_by_operation),
            "fab_suitability_map": [
                {
                    "kind": "region",
                    "suitability": InlineDict(self.region_suitability),
                },
                {
                    "kind": "customer_company",
                    "suitability": InlineDict(self.customer_suitability),
                },
            ],
            "unavailable_dates": unavailable,
            "affinity": InlineList(self.affinity),
        }


def _parse_date(s: str) -> date:
    y, m, d = s.split("/")
    return date(int(y), int(m), int(d))


def _generate_unavailable_dates() -> list[str]:
    start = _parse_date(unavailable_date_range_start)
    end = _parse_date(unavailable_date_range_end)

    count = random.choices(
        range(unavailable_max_dates + 1),
        unavailable_count_weights,
        k=1,
    )[0]

    if count == 0:
        return []

    all_dates: list[date] = []
    d = start
    while d <= end:
        all_dates.append(d)
        d += timedelta(days=1)

    count = min(count, len(all_dates))
    chosen = sorted(random.sample(all_dates, count))
    return [d.strftime("%Y/%m/%d") for d in chosen]


def create_affinity_tags() -> list[dict]:
    tags = []
    for i in range(affinity_group_num):
        weight = random.choices(affinity_weight_options, affinity_weight_chances, k=1)[0]
        tags.append({"id": f"a{i + 1}", "weight": weight})
    return tags


def assign_affinity_tags(worker_list: list[WorkerDef], affinity_tags: list[dict]) -> None:
    for tag in affinity_tags:
        tag_id = tag["id"]
        size = random.randint(
            affinity_group_size[0],
            min(affinity_group_size[1], len(worker_list)),
        )
        for worker in random.sample(worker_list, size):
            if tag_id not in worker.affinity:
                worker.affinity.append(tag_id)


def create_worker_list(
    worker_num: int,
    worker_company_map: dict[str, CompanyDef],
    workflow: WorkflowDef,
    region_map: dict[str, RegionDef],
    customer_company_map: dict[str, CompanyDef],
) -> list[WorkerDef]:
    skill_list: list[OperationDef] = [
        o for p in workflow.phase_list for o in p.operation_list
    ]
    skill_num_range = (3, 6)

    company_list = list(worker_company_map.values())
    region_ids = list(region_map.keys())
    customer_ids = list(customer_company_map.keys())

    worker_list: list[WorkerDef] = []
    for i in range(worker_num):
        wid = f"w{i + 1}"
        name = chr(ord("A") + i // 26) + chr(ord("A") + i % 26)

        skill_num = random.randint(skill_num_range[0], skill_num_range[1])
        skill_index_list = sorted(random.sample(range(len(skill_list)), skill_num))
        skills = [skill_list[idx] for idx in skill_index_list]
        skill_levels = random.choices(
            skill_level_list,
            skill_level_weights,
            k=len(skills),
        )

        # each skilled operation gets regular or spot based on configured chance
        worker_type_by_operation: dict[str, str] = {
            op.id: random.choices(
                ["regular", "spot"],
                [worker_type_regular_chance, 1.0 - worker_type_regular_chance],
                k=1,
            )[0]
            for op in skills
        }

        company = random.choice(company_list)
        is_manager = random.choices(
            [True, False],
            [manager_rate, 1.0 - manager_rate],
            k=1,
        )[0]

        region_suitability: dict[str, int] = {
            rid: random.choices(
                region_suitability_list,
                region_suitability_weights,
                k=1,
            )[0]
            for rid in region_ids
        }

        customer_suitability: dict[str, int] = {
            cid: random.choices(
                customer_suitability_list,
                customer_suitability_weights,
                k=1,
            )[0]
            for cid in customer_ids
        }

        unavailable_dates = _generate_unavailable_dates()

        worker = WorkerDef(
            id=wid,
            name=name,
            company=company,
            is_manager=is_manager,
            role="",
            skill_map=list(zip(skills, skill_levels)),
            worker_type_by_operation=worker_type_by_operation,
            region_suitability=region_suitability,
            customer_suitability=customer_suitability,
            unavailable_dates=unavailable_dates,
            affinity=[],
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
    affinity_tags: list[dict],
    transit_day_map: list[dict],
):
    env_data: dict[str, object] = {}

    env_data["workflow_list"] = [workflow.to_dict()]
    env_data["fab_list"] = [f.to_dict() for f in fab_map.values()]
    env_data["region_list"] = [r.to_dict() for r in region_map.values()]
    env_data["customer_company_list"] = [
        c.to_customer_dict() for c in customer_company_map.values()
    ]
    env_data["worker_company_list"] = [
        c.to_worker_dict() for c in worker_company_map.values()
    ]
    env_data["transite_day_map"] = transit_day_map
    env_data["affinity_tag"] = affinity_tags
    env_data["worker_list"] = [w.to_dict() for w in worker_list]

    root_data = {
        "environment": env_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(root_data, f, allow_unicode=True, sort_keys=False)


# ---------------- Schedule generation ----------------

WEEKDAY_TAGS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def is_region_holiday(day: date, region: RegionDef) -> bool:
    weekday_tag = WEEKDAY_TAGS[day.weekday()]
    if weekday_tag in region.weekly_weekdays:
        return True

    day_str = day.strftime("%Y/%m/%d")
    if day_str in region.single_days:
        return True

    return False


def is_holiday(day: date, fab: FabDef | None, region_map: dict[str, RegionDef] | None = None) -> bool:
    if fab is not None:
        return is_region_holiday(day, fab.region)

    if region_map is not None and len(region_map) > 0:
        return all(is_region_holiday(day, r) for r in region_map.values())

    return False


def add_working_days(day: date, days: int, fab: FabDef | None) -> date:
    current = day
    remaining = days
    while remaining > 0:
        current = current + timedelta(days=1)
        if not is_holiday(current, fab):
            remaining -= 1
    return current


def to_eq_operation_dict(
    operation: OperationDef,
    workload_days: int,
    eq_id: str,
) -> dict:
    op_task_id = f"{eq_id}{operation.id}"

    # convert days to hours if configured
    if workload_format == "hours":
        workload_key = "workload_hours"
        workload_value = workload_days * workload_units
    else:
        workload_key = "workload_days"
        workload_value = workload_days

    rec_min, rec_max = random.choices(
        recommends_worker_options,
        recommends_worker_weights,
        k=1,
    )[0]

    return {
        "id": op_task_id,
        "name": operation.name,
        "operation": operation.id,
        workload_key: workload_value,
        "recommends_worker_min": rec_min,
        "recommends_worker_max": rec_max,
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
        "operation_task_list": operation_task_list,
    }


def to_eq_dict(
    index: int,
    workflow: WorkflowDef,
    fab: FabDef,
    start_day: date,
    worklength: list[tuple[int, list[int]]],
) -> tuple[dict, date]:
    eq_id = f"e{index + 1}"
    name = f"SU {1000 + index + 1}A"

    phase_task_list: list[dict] = []

    module_start = start_day
    while is_holiday(module_start, fab):
        module_start = module_start + timedelta(days=1)

    cumulative_days = 0
    final_end_day = module_start

    for i, phase in enumerate(workflow.phase_list):
        phase_length_days = worklength[i][0]
        op_worklengths = worklength[i][1]

        cumulative_days += phase_length_days

        if cumulative_days <= 1:
            phase_end = module_start
        else:
            phase_end = add_working_days(module_start, cumulative_days - 1, fab)

        phase_dict = to_eq_phase_dict(
            phase=phase,
            operation_worklengths=op_worklengths,
            start_day=module_start,
            end_day=phase_end,
            eq_id=eq_id,
        )
        phase_task_list.append(phase_dict)

        if phase_end > final_end_day:
            final_end_day = phase_end

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
    return (
        [normal_worklength, vip_worklength],
        [0.8, 0.2],
    )


def create_equipment_list(
    workflow: WorkflowDef,
    fab_map: dict[str, FabDef],
    region_map: dict[str, RegionDef],
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
        while eq_point >= 1.0 and eq_count < eq_num:
            worklength = random.choices(worklength_list, worklength_weights, k=1)[0]
            fab = random.choice(fab_list)

            eq_start_day = current_day
            while is_holiday(eq_start_day, fab, region_map):
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

        while True:
            current_day = current_day + timedelta(days=1)
            if not is_holiday(current_day, None, region_map):
                break

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

    transit_day_map = create_transit_day_map(region_map)
    affinity_tags = create_affinity_tags()
    worker_list = create_worker_list(
        WORKER_NUM,
        worker_company_map,
        workflow,
        region_map,
        customer_company_map,
    )
    assign_affinity_tags(worker_list, affinity_tags)

    write_environment_yaml(
        "EnvConfig.yaml",
        workflow,
        fab_map,
        region_map,
        customer_company_map,
        worker_company_map,
        worker_list,
        affinity_tags,
        transit_day_map,
    )

    # ----- Schedule / equipment -----
    random.seed(EQ_SEED)

    equipment_list, end_day = create_equipment_list(
        workflow,
        fab_map,
        region_map,
        EQ_PER_DAYS,
        EQ_PER_DAYS_SIGMA,
        EQ_NUM,
        START_DAY,
    )

    write_schedule_yaml(
        "Schedule.yaml",
        equipment_list,
        START_DAY,
        end_day + timedelta(days=3),
    )


if __name__ == "__main__":
    main()
