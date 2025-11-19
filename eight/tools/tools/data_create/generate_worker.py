#!/usr/bin/env python3
# generate_worker.py
# Overwrite environment.worker_list in EnvConfig.yaml
# python generate_worker.py --env EnvConfig.yaml
# python generate_worker.py --env EnvConfig.yaml --n 120 --mgr_pct 50 \
#   --skill_min 4 --skill_max 6 \
#   --levels 1,2,3,4,5 --level_weights 0.05,0.15,0.5,0.2,0.1 \
#   --unavail_emp_pct 30 --unavail_max 3 \
#   --schedule Schedule.yaml
import argparse
import random
import string
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# -------------------- DEFAULTS (edit once, run many) --------------------
DEFAULT_NUM_EMPLOYEES = 120
DEFAULT_MANAGER_PERCENT = 100

DEFAULT_SKILL_COUNT_MIN = 4
DEFAULT_SKILL_COUNT_MAX = 6

# available levels and their global distribution
# DEFAULT_LEVELS = [1, 2, 3, 4, 5]
# DEFAULT_LEVEL_WEIGHTS = [10, 20, 40, 20, 10]  # sum ~= 100
DEFAULT_LEVELS = [1]
DEFAULT_LEVEL_WEIGHTS = [100]  # sum ~= 100

# how many regions per employee (cardinality) -> probability
# keys are integers (0 means none), values are probabilities (sum ~= 1.0)
DEFAULT_REGION_CARDINALITY_WEIGHTS = {0: 1}

# percentage of employees who will have unavailable dates and max count per such employee
DEFAULT_UNAVAIL_EMP_PERCENT = 0
DEFAULT_UNAVAIL_MAX_PER_EMP = 1

# Plan range for generating unavailable_dates (used only if you want random personal off-days)
DEFAULT_PLAN_START = "2025/09/01"
DEFAULT_PLAN_END = "2025/01/10"

# random seed (None for non-deterministic)
DEFAULT_SEED = 42
# ------------------------------------------------------------------------

# --- helpers: inline (flow) YAML styles ---
class FlowDict(dict):
    """Marker dict that should be dumped in flow style {a:1, b:2}."""
    pass

class FlowList(list):
    """Marker list that should be dumped in flow style [a, b, c]."""
    pass

def _flow_map_representer(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

def _flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowDict, _flow_map_representer)
yaml.add_representer(FlowList, _flow_list_representer)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="Path to EnvConfig.yaml to overwrite")
    p.add_argument("--schedule", required=False, help="Optional Schedule.yaml (to get plan_range for unavailable dates)")

    p.add_argument("--num", type=int, default=DEFAULT_NUM_EMPLOYEES)
    p.add_argument("--mgr", type=float, default=DEFAULT_MANAGER_PERCENT, help="percent of managers")

    p.add_argument("--skill-min", type=int, default=DEFAULT_SKILL_COUNT_MIN)
    p.add_argument("--skill-max", type=int, default=DEFAULT_SKILL_COUNT_MAX)
    p.add_argument("--levels", type=str, default=",".join(map(str, DEFAULT_LEVELS)))
    p.add_argument("--level-weights", type=str, default=",".join(map(str, DEFAULT_LEVEL_WEIGHTS)))

    p.add_argument("--region-cardinality", type=str,
                   default=",".join(f"{k}:{v}" for k, v in DEFAULT_REGION_CARDINALITY_WEIGHTS.items()))

    p.add_argument("--unavail-emp", type=float, default=DEFAULT_UNAVAIL_EMP_PERCENT)
    p.add_argument("--unavail-max", type=int, default=DEFAULT_UNAVAIL_MAX_PER_EMP)

    p.add_argument("--plan-start", type=str, default=DEFAULT_PLAN_START)
    p.add_argument("--plan-end", type=str, default=DEFAULT_PLAN_END)

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path, data):
    # BLOCK for everything, but FlowDict/FlowList stay inline because of custom representers
    dumper = yaml.Dumper
    dumper.ignore_aliases = lambda self, data: True  # avoid anchors
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, Dumper=dumper, allow_unicode=True, sort_keys=False, default_flow_style=False)


def ymd_iter(start_str, end_str):
    df = "%Y/%m/%d"
    start = datetime.strptime(start_str.replace("-", "/"), df).date()
    end = datetime.strptime(end_str.replace("-", "/"), df).date()
    cur = start
    while cur <= end:
        yield cur.strftime(df)
        cur += timedelta(days=1)

def alpha_names(n):
    # AA, AB, ..., AZ, BA, BB, ..., ZZ, AAA, AAB, ...
    letters = string.ascii_uppercase
    out = []
    L = 2  # start with 2-letter codes
    while len(out) < n:
        base = 26 ** (L - 1)
        for idx in range(26 ** L):
            # build an L-letter string in base-26 (A=0..Z=25)
            name = ''.join(letters[(idx // (26 ** p)) % 26] for p in reversed(range(L)))
            out.append(name)
            if len(out) == n:
                break
        L += 1
    return out


def weighted_choice(items, weights):
    # items: list[T], weights: list[float] (not necessarily normalized)
    total = sum(weights)
    r = random.random() * total
    c = 0.0
    for it, w in zip(items, weights):
        c += w
        if r <= c:
            return it
    return items[-1]


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    root = load_yaml(args.env)
    env = root.get("environment", root)

    # collect operations from workflow_list (p#o# ids)
    op_ids = []
    for wf in env.get("workflow_list", []):
        for ph in wf.get("phase_list", []):
            for op in ph.get("operation_list", []):
                op_ids.append(str(op.get("id")))
    op_ids = [x for x in op_ids if x]  # non-empty

    if not op_ids:
        raise SystemExit("No operation ids found under environment.workflow_list.*.phase_list.*.operation_list[].id")

    # regions (by id)
    region_ids = [str(r.get("id")) for r in env.get("region_list", []) if r.get("id")]

    # worker companies (by id) -> random select per worker
    company_ids = [str(c.get("id")) for c in env.get("worker_company_list", []) if c.get("id")]
    if not company_ids:
        # fallback: keep compatibility if user forgot to define companies
        company_ids = ["c2"]

    # parse region cardinality weights (e.g., "0:0.2,1:0.6,2:0.2")
    card_kv = {}
    for tok in str(args.region_cardinality).split(","):
        if not tok:
            continue
        k, v = tok.split(":")
        card_kv[int(k.strip())] = float(v.strip())

    # parse levels & weights
    levels = [int(x.strip()) for x in str(args.levels).split(",") if x.strip()]
    lvl_weights = [float(x.strip()) for x in str(args.level_weights).split(",") if x.strip()]
    if len(levels) != len(lvl_weights):
        raise SystemExit("--levels and --level-weights lengths differ")

    # plan range for generating personal unavailable dates
    plan_start = args.plan_start
    plan_end = args.plan_end
    # if schedule provided, try to read plan_range from it
    if args.schedule:
        try:
            sched = load_yaml(args.schedule)
            s = sched.get("schedule", sched)
            pr = s.get("plan_range", {})
            ps = pr.get("start_date")
            pe = pr.get("end_date")
            if ps and pe:
                plan_start = str(ps).replace("-", "/")
                plan_end = str(pe).replace("-", "/")
        except Exception:
            pass

    # build a pool of dates in range for random unavailability
    all_dates = list(ymd_iter(plan_start, plan_end))

    num = int(args.num)
    mgr_count = round(num * float(args.mgr) / 100.0)

    names = alpha_names(num)
    ids = [f"w{i+1}" for i in range(num)]

    # --- choose managers randomly ---
    manager_indices = set(random.sample(range(num), mgr_count)) if mgr_count > 0 else set()

    # Distribute skills roughly evenly across operations
    # (so ops don't end up extremely unbalanced)
    desired_per_op = defaultdict(int)

    # total skills to draw = sum per-employee chosen skill counts
    total_skill_slots = 0
    per_emp_skill_counts = []
    for _ in range(num):
        k = random.randint(int(args.skill_min), int(args.skill_max))
        per_emp_skill_counts.append(k)
        total_skill_slots += k

    # target per-op count
    base = total_skill_slots // len(op_ids)
    remainder = total_skill_slots % len(op_ids)
    for i, op in enumerate(op_ids):
        desired_per_op[op] = base + (1 if i < remainder else 0)

    # build a working bag we can pop from, respecting balance
    op_bag = []
    for op, cnt in desired_per_op.items():
        op_bag.extend([op] * cnt)
    random.shuffle(op_bag)

    workers = []
    for i in range(num):
        wid = ids[i]
        nm = names[i]

        is_manager = (i in manager_indices)

        # random worker_company from list provided
        worker_company = random.choice(company_ids)

        # choose how many regions for this employee, then pick that many distinct regions
        if region_ids:
            card_choices, card_weights = zip(*sorted(card_kv.items()))
            card = weighted_choice(card_choices, card_weights)
            card = max(0, min(card, len(region_ids)))
            chosen_regions = sorted(random.sample(region_ids, card)) if card > 0 else []
        else:
            chosen_regions = []

        # --- draw exactly K DISTINCT operations (respect min/max, cap by available ops) ---
        K_target = per_emp_skill_counts[i]
        K_target = max(int(args.skill_min), min(int(args.skill_max), K_target))
        K_target = min(K_target, len(op_ids))  # cannot exceed number of distinct ops

        chosen_ops_set = set()
        attempts = 0
        max_attempts = K_target * 10
        while len(chosen_ops_set) < K_target and attempts < max_attempts:
            cand = op_bag.pop() if op_bag else random.choice(op_ids)
            if cand not in chosen_ops_set:
                chosen_ops_set.add(cand)
            attempts += 1

        if len(chosen_ops_set) < K_target:
            remaining_ops = [op for op in op_ids if op not in chosen_ops_set]
            random.shuffle(remaining_ops)
            need = K_target - len(chosen_ops_set)
            chosen_ops_set.update(remaining_ops[:need])

        chosen_ops = list(chosen_ops_set)
        # --- end distinct K selection ---

        # assign a level to each chosen op, using the global level distribution
        skill_map = FlowDict()
        for op in chosen_ops:
            lvl = weighted_choice(levels, lvl_weights)
            skill_map[op] = int(lvl)

        # random unavailable dates (for a fraction of employees)
        unavailable_dates = []
        if all_dates and (random.random() * 100.0 < float(args.unavail_emp)):
            kmax = max(0, int(args.unavail_max))
            k = random.randint(1, kmax) if kmax > 0 else 0
            if k > 0:
                unavailable_dates = sorted(random.sample(all_dates, min(k, len(all_dates))))

        worker = {
            "id": wid,
            "name": nm,
            "worker_company": worker_company,
            "is_manager": bool(is_manager),
            # INLINE for boss: {p1o1: 3, p2o2: 1, ...}
            "skill_map": skill_map,
            # INLINE list styles:
            "fab_suitability_map": FlowList(chosen_regions),
            "unavailable_dates": FlowList(unavailable_dates),
        }
        workers.append(worker)

    # write back into env
    env["worker_list"] = workers
    if "environment" in root:
        root["environment"] = env

    save_yaml(args.env, root)
    print(f"Generated {num} workers into {args.env}")
    # quick stats:
    c_by_company = Counter(w["worker_company"] for w in workers)
    print("Worker companies:", dict(c_by_company))
    c_by_skill = Counter()
    for w in workers:
        for op in w["skill_map"].keys():
            c_by_skill[op] += 1
    print("Skill coverage (#employees per op):", dict(c_by_skill))


if __name__ == "__main__":
    main()
