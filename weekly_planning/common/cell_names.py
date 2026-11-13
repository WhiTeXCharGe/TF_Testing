"""
The name written in the cells of a module ("cell name"): FlexibleSchedule.yaml keeps a `cell_name` in every assignment row,
3-2 writes it, 3-3 puts it into the workbook, 2-2 compares it to see whether a planner renamed a bar.
"""
from __future__ import annotations

import re
import unicodedata


# FlexibleSchedule.yaml has a `cell_name` in every assignment row of a module: the text of the bar in SU_Others.  It is
# written by 3-2 (from the solver result), used by 3-3 to write the cells and checked by 2-2 to see whether the planner
# changed the text of a bar (a changed text makes the module Fixed, like a moved bar).
DEFAULT_CELL_NAME_TEMPLATE = "{customer}{fab}_{code}_新規"


def cell_name_key(text):
    """Comparison key of a cell text: NFKC, upper case, no whitespace (so only a real change of the text counts)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).upper()


def module_name_info(task, env):
    """{"fab", "customer", "description"} names of a workflow task (task.fab id -> EnvConfig fab_list / customer_company_list)."""
    fabs = {f["id"]: f for f in (env or {}).get("fab_list") or []}
    customers = {c["id"]: c.get("name") for c in (env or {}).get("customer_company_list") or []}
    fab = fabs.get(task.get("fab")) or {}
    return {"fab": fab.get("name") or "", "customer": customers.get(fab.get("customer_company")) or "",
            "description": task.get("description") or ""}


def module_cell_name(code, info, template=DEFAULT_CELL_NAME_TEMPLATE):
    """Name for the cells of a module that has none, e.g. '{customer}{fab}_{code}_新規' -> 'CustAFabA_530N00000A_新規'."""
    text = template.format(code=code, fab=info.get("fab", ""), customer=info.get("customer", ""),
                           description=info.get("description", ""))
    text = re.sub(r"_{2,}", "_", text).strip("_ ")
    return text or code
