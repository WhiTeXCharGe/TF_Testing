"""
Skill text of a worker (role + skill level) -> the kinds of work the worker can do, and the worker status -> worker type.

The skill is kept per WORK NAME (Mech, Elec, QC), not per operation id: the caller gives the level to every operation of that
name in the workflows of the EnvConfig (see tool 2-2-3).
"""
from __future__ import annotations

import re

from .text import clean_text


# role token of the skill sheet -> work name of the workflows ("Q" is a lone-Q typo of QC seen once)
ROLE_TOKEN_WORK = {"M": "Mech", "E": "Elec", "QC": "QC", "Q": "QC"}


WORK_NAMES = ("Mech", "Elec", "QC")


STATUS_TO_TYPE = {"PRIMARY": "regular"}  # everything else (Secondary / blank) -> spot


# level given to a worker who has no skill level anywhere and is only known by the 担当職種 (role) of SU_Others:
# the lowest level the skill sheet uses (1 to 3), so the worker can be planned but is not preferred
DEFAULT_ROLE_LEVEL = 1


def split_slash(s: str):
    return [t.strip() for t in re.split(r"[/／]", s) if t.strip()]


def role_skills(role_text, level=DEFAULT_ROLE_LEVEL):
    """{work name: level} from a 担当職種 text alone ("M/QC" -> Mech and QC at `level`); roles that are not a kind of work
    in the workflows (搬送, 通訳 ...) give nothing. Ordered Mech, Elec, QC."""
    works = {ROLE_TOKEN_WORK.get(t.upper()) for t in split_slash(clean_text(role_text or ""))}
    return {work: level for work in WORK_NAMES if work in works} if level and level > 0 else {}


def parse_worker_skills(role_text: str, skill_raw: str):
    """
    Returns (skill_map, excluded_reason). skill_map is {work name: level} (Mech / Elec / QC) with only
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
        work = ROLE_TOKEN_WORK.get(role)
        if work:
            skill_map[work] = max(skill_map.get(work, 0), lv)
    skill_map = {work: skill_map[work] for work in WORK_NAMES if work in skill_map}
    return skill_map, None