"""
Skill text of a worker (role + skill level) -> the operations the worker can do, and the worker status -> worker type.
"""
from __future__ import annotations

import re

from .text import clean_text


# role token -> operations it qualifies for ("Q" is a lone-Q typo of QC seen once)
ROLE_TOKEN_OPS = {"M": ("p2o1",), "E": ("p2o2",), "QC": ("p3o1", "p4o1"), "Q": ("p3o1", "p4o1")}


TASK_OPS = ("p2o1", "p2o2", "p3o1", "p4o1")


STATUS_TO_TYPE = {"PRIMARY": "regular"}  # everything else (Secondary / blank) -> spot


def split_slash(s: str):
    return [t.strip() for t in re.split(r"[/／]", s) if t.strip()]


def parse_worker_skills(role_text: str, skill_raw: str):
    """
    Returns (skill_map, excluded_reason). skill_map is {op: level} with only
    nonzero levels; excluded_reason is None for planning targets.

      - 担当職種 blank                       -> excluded
      - スキルレベル 0 (or blank/#N/A)        -> excluded
      - "M/QC" + "2/3"                      -> M=2, QC=3
      - "M/QC" + "3"                        -> M=3, QC=3
    """
    roles = [r.upper() for r in split_slash(clean_text(role_text or ""))]
    if not roles:
        return {}, "担当職種 blank"

    levels = []
    for tok in split_slash(skill_raw or ""):
        try:
            levels.append(int(float(tok)))
        except ValueError:
            levels.append(None)
    if not levels:
        return {}, "スキルレベル blank"
    if all(lv is None for lv in levels):
        return {}, f"スキルレベル unreadable ({skill_raw!r})"
    if all((lv or 0) == 0 for lv in levels):
        return {}, "スキルレベル 0"

    if len(levels) == 1:
        per_role = [levels[0]] * len(roles)
    elif len(levels) == len(roles):
        per_role = levels
    else:
        # count mismatch (not seen in the data so far): pair by position,
        # reuse the last level for any extra roles
        per_role = [levels[min(i, len(levels) - 1)] for i in range(len(roles))]

    skill_map = {}
    for role, lv in zip(roles, per_role):
        if not lv or lv <= 0:
            continue
        for op in ROLE_TOKEN_OPS.get(role, ()):
            skill_map[op] = max(skill_map.get(op, 0), lv)
    skill_map = {op: skill_map[op] for op in TASK_OPS if op in skill_map}
    return skill_map, None
