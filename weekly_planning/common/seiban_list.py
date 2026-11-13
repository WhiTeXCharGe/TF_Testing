"""
The seiban list workbook (read by tool 2-4).  Nothing is decided here: the workbook is turned into plain data, and everything
that looks wrong is kept as text in `problems`, so the caller can skip a row and put the reason into its report.

Sheets (names are the defaults of the constants below):
  計画対象          the rows to handle (one seiban per row; the colour of the seiban cell says add / keep / delete)
  搬入先テーブル     delivery sites: customer, Fab, Region (used to fill an empty Region)
  工程テンプレート<N> one sheet per process template: process number, work number, work name, man-hours, min / max workers

The workbook is opened read-only, so the big master-list sheet of the file is never loaded.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from openpyxl import load_workbook

from .dates import as_timestamp
from .text import match_key, tool_code

PLAN_SHEET = "計画対象"
SITE_SHEET = "搬入先テーブル"
TEMPLATE_PREFIX = "工程テンプレート"

# what the colour of a row means
ADD, KEEP, DELETE, UNKNOWN = "add", "keep", "delete", "unknown"
WHITE_RGB, GREY_RGB, PINK_RGB = "FFFFFF", "D9D9D9", "FFDCE6"
COLOUR_KIND = {WHITE_RGB: ADD, GREY_RGB: KEEP, PINK_RGB: DELETE}

BLANK_RUN_LIMIT = 100      # the sheet stops after this many empty rows in a row (formatting can make a sheet "very long")

_TEMPLATE_NO = re.compile(re.escape(TEMPLATE_PREFIX) + r"\s*(\d+)")
_PROCESS_HEADER = re.compile(r"工程(\d+)_(開始可能日|終了期限日)")


class SeibanListError(ValueError):
    """The workbook cannot be used at all (a sheet or a column that is needed is missing)."""


@dataclass
class PlanRow:
    excel_row: int
    kind: str                        # ADD | KEEP | DELETE | UNKNOWN
    colour: str                      # "none", "FFFFFF" ... or a description of a colour that is not known
    code: str | None                 # the module code (seiban), None when the cell is not one
    raw_code: str
    customer: str                    # trimmed text of the sheet, "" when empty
    fab: str
    region: str
    template: int | None             # N of "工程テンプレートN"
    template_text: str
    phases: dict = field(default_factory=dict)      # {N: (start 'YYYY/MM/DD' | None, end 'YYYY/MM/DD' | None)} for every process N of the sheet
    problems: list = field(default_factory=list)


@dataclass
class Site:
    site: str
    customer: str
    fab: str
    region: str


@dataclass
class TemplateOp:
    phase: int
    op: int
    name: str                        # normalised work name ("Q/C" -> "QC")
    hours: float | int               # 工数
    min_workers: int
    max_workers: int


@dataclass
class Template:
    number: int
    sheet: str
    ops: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @property
    def valid(self):
        return not self.problems and bool(self.ops)

    def phases(self):
        return sorted({o.phase for o in self.ops})

    def ops_of(self, phase):
        return sorted((o for o in self.ops if o.phase == phase), key=lambda o: o.op)


@dataclass
class SeibanList:
    rows: list
    sites: list
    templates: dict                  # {N: Template}
    warnings: list                   # about the workbook as a whole (duplicate sites, a missing sheet ...)

    def __post_init__(self):
        self._site_index = {}
        for s in self.sites:
            self._site_index.setdefault((match_key(s.customer), match_key(s.fab)), s)

    def rows_of(self, kind):
        return [r for r in self.rows if r.kind == kind]

    def site_region(self, customer, fab):
        """Region of a delivery site (same customer and fab by name key), or None."""
        s = self._site_index.get((match_key(customer), match_key(fab)))
        return s.region if s and s.region else None


# ----------------------------------------------------------------------------- small readers

def _text(v):
    """Trimmed text of a cell value ('' for an empty cell, whole numbers without '.0')."""
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _number(v):
    """int / float of a cell value, None when it is not a number."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else v
    try:
        f = float(unicodedata.normalize("NFKC", str(v)).strip())
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def _date(v):
    """('YYYY/MM/DD' | None, problem | None).  An empty cell is None without a problem."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None, None
    ts = as_timestamp(v)
    if ts is None:
        return None, f"not a date: {v!r}"
    return ts.strftime("%Y/%m/%d"), None


def normalise_work_name(text):
    """Work name as the EnvConfig writes it: NFKC, no spaces, 'Q/C' -> 'QC'."""
    name = re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))
    return "QC" if name.upper() == "Q/C" else name


def _headers(first_row):
    """{match_key(header): column index} of the first row."""
    out = {}
    for i, c in enumerate(first_row):
        v = getattr(c, "value", c)
        if v is not None and str(v).strip():
            out.setdefault(match_key(v), i)
    return out


def _need(headers, names, where):
    missing = [n for n in names if match_key(n) not in headers]
    if missing:
        raise SeibanListError(f"{where}: column {', '.join(missing)} not found in the first row")
    return {n: headers[match_key(n)] for n in names}


def _cell(row, idx):
    return row[idx] if idx is not None and idx < len(row) else None


def _value(row, idx):
    c = _cell(row, idx)
    return getattr(c, "value", None) if c is not None else None


def row_kind(cell):
    """(kind, colour) from the fill of a cell: no fill / white = add, grey = keep, pink = delete, anything else = unknown."""
    fill = getattr(cell, "fill", None)
    if fill is None or fill.fill_type in (None, "none"):
        return ADD, "none"
    if fill.fill_type != "solid":
        return UNKNOWN, f"pattern:{fill.fill_type}"
    c = fill.fgColor
    if c.type == "rgb" and isinstance(c.rgb, str):
        rgb = c.rgb[-6:].upper()
        return COLOUR_KIND.get(rgb, UNKNOWN), rgb
    if c.type == "theme":
        if c.theme == 0 and not c.tint:
            return ADD, "theme:0"          # the white of the theme
        return UNKNOWN, f"theme:{c.theme}"
    return UNKNOWN, f"{c.type}:{getattr(c, c.type, '')}"


# ----------------------------------------------------------------------------- the three kinds of sheets

def _read_plan(ws, blank_run_limit):
    rows = ws.iter_rows()
    first = next(rows, None)
    if first is None:
        raise SeibanListError(f"sheet {ws.title}: empty")
    headers = _headers(first)
    col = _need(headers, ["製番", "顧客", "Fab", "Region", "工程選択"], f"sheet {ws.title}")
    process_cols = {}
    for key, idx in headers.items():
        m = _PROCESS_HEADER.fullmatch(unicodedata.normalize("NFKC", key))
        if m:
            process_cols.setdefault(int(m.group(1)), {})["start" if m.group(2) == "開始可能日" else "end"] = idx
    if not process_cols:
        raise SeibanListError(f"sheet {ws.title}: no process date columns (工程<N>_開始可能日 / 工程<N>_終了期限日)")

    out, blank_run = [], 0
    for excel_row, row in enumerate(rows, start=2):
        code_cell = _cell(row, col["製番"])
        values = [getattr(c, "value", None) for c in row]
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in values):
            blank_run += 1
            if blank_run >= blank_run_limit:
                break
            continue
        blank_run = 0
        problems = []
        kind, colour = row_kind(code_cell)
        if kind == UNKNOWN:
            problems.append(f"row colour {colour} is not white, grey or pink")
        raw_code = _text(getattr(code_cell, "value", None))
        code = tool_code(raw_code)
        if not raw_code:
            problems.append("seiban is empty")
        elif not code:
            problems.append(f"seiban {raw_code!r} is not a module code")
        template_text = _text(_value(row, col["工程選択"]))
        m = _TEMPLATE_NO.search(unicodedata.normalize("NFKC", template_text))
        template = int(m.group(1)) if m else None
        if not template_text:
            problems.append("no process template chosen (工程選択 is empty)")
        elif template is None:
            problems.append(f"process template {template_text!r} is not readable")
        phases = {}
        for n in sorted(process_cols):
            start, p1 = _date(_value(row, process_cols[n].get("start")))
            end, p2 = _date(_value(row, process_cols[n].get("end")))
            if p1:
                problems.append(f"process {n} start: {p1}")
            if p2:
                problems.append(f"process {n} end: {p2}")
            phases[n] = (start, end)
        out.append(PlanRow(excel_row, kind, colour, code, raw_code,
                           _text(_value(row, col["顧客"])), _text(_value(row, col["Fab"])), _text(_value(row, col["Region"])),
                           template, template_text, phases, problems))
    return out


def _read_sites(ws, warnings):
    rows = ws.iter_rows()
    first = next(rows, None)
    headers = _headers(first) if first else {}
    try:
        col = _need(headers, ["顧客", "Fab", "Region"], f"sheet {ws.title}")
    except SeibanListError as e:
        warnings.append(str(e))
        return []
    site_col = headers.get(match_key("搬入先"))
    sites, seen = [], {}
    for row in rows:
        customer, fab, region = _text(_value(row, col["顧客"])), _text(_value(row, col["Fab"])), _text(_value(row, col["Region"]))
        site = _text(_value(row, site_col))
        if not (site or customer or fab or region):
            continue
        key = (match_key(customer), match_key(fab))
        if key in seen:
            if match_key(seen[key].region) != match_key(region):
                warnings.append(f"sheet {ws.title}: {customer} / {fab} appears twice with different regions "
                                f"({seen[key].region!r}, {region!r}); the first is used")
            continue
        seen[key] = Site(site, customer, fab, region)
        sites.append(seen[key])
    return sites


def _read_template(ws, number):
    t = Template(number, ws.title)
    rows = ws.iter_rows()
    first = next(rows, None)
    headers = _headers(first) if first else {}
    try:
        col = _need(headers, ["工程番号", "作業番号", "作業名", "工数", "最小人数", "最大人数"], f"sheet {ws.title}")
    except SeibanListError as e:
        t.problems.append(str(e))
        return t
    seen = set()
    for excel_row, row in enumerate(rows, start=2):
        vals = {k: _value(row, i) for k, i in col.items()}
        if all(v is None or (isinstance(v, str) and not v.strip()) for v in vals.values()):
            continue
        phase, op = _number(vals["工程番号"]), _number(vals["作業番号"])
        hours, lo, hi = _number(vals["工数"]), _number(vals["最小人数"]), _number(vals["最大人数"])
        name = normalise_work_name(_text(vals["作業名"]))
        bad = [label for label, v in (("工程番号", phase), ("作業番号", op), ("工数", hours), ("最小人数", lo), ("最大人数", hi))
               if not isinstance(v, (int, float)) or v < 0]
        if not name:
            bad.append("作業名")
        if bad:
            t.problems.append(f"row {excel_row}: {', '.join(bad)} is empty or not a number")
            continue
        if not (isinstance(phase, int) and isinstance(op, int) and phase >= 1 and op >= 1):
            t.problems.append(f"row {excel_row}: 工程番号 and 作業番号 must be whole numbers from 1")
            continue
        if not (isinstance(lo, int) and isinstance(hi, int) and lo >= 1 and hi >= lo):
            t.problems.append(f"row {excel_row}: workers must be whole numbers with 1 <= min <= max (got {lo}, {hi})")
            continue
        if (phase, op) in seen:
            t.problems.append(f"row {excel_row}: process {phase} work {op} appears twice")
            continue
        seen.add((phase, op))
        t.ops.append(TemplateOp(phase, op, name, hours, lo, hi))
    if not t.ops and not t.problems:
        t.problems.append(f"sheet {ws.title}: no operations")
    return t


# ----------------------------------------------------------------------------- entry point

def read_seiban_list(path, plan_sheet=PLAN_SHEET, blank_run_limit=BLANK_RUN_LIMIT):
    """Reads the workbook.  Raises SeibanListError when the plan sheet or one of its columns is missing."""
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if plan_sheet not in wb.sheetnames:
            raise SeibanListError(f"sheet {plan_sheet!r} not found in {path}; available: {wb.sheetnames}")
        warnings = []
        rows = _read_plan(wb[plan_sheet], blank_run_limit)
        if SITE_SHEET in wb.sheetnames:
            sites = _read_sites(wb[SITE_SHEET], warnings)
        else:
            sites = []
            warnings.append(f"sheet {SITE_SHEET!r} not found: an empty Region cannot be filled from it")
        templates = {}
        for name in wb.sheetnames:
            m = _TEMPLATE_NO.fullmatch(unicodedata.normalize("NFKC", name).strip())
            if m:
                n = int(m.group(1))
                if n in templates:
                    warnings.append(f"sheets {templates[n].sheet!r} and {name!r} are the same template number; the first is used")
                else:
                    templates[n] = _read_template(wb[name], n)
        if not templates:
            warnings.append(f"no sheet named {TEMPLATE_PREFIX}<N> found")
        return SeibanList(rows, sites, templates, warnings)
    finally:
        wb.close()
