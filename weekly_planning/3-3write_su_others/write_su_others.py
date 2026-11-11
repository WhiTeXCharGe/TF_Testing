#!/usr/bin/env python3
"""
write_su_others.py  --  weekly planning tool "3-3": write the flexible schedule back into SU_Others.

Takes the schedule of the flexible modules (a FlexibleSchedule.yaml, a decoder7 Schedule.yaml or a Timefold result
Schedule.yaml) and the old SU_Others workbook and writes a NEW workbook in which the cells of those modules follow the
schedule.  Nothing else changes:

  * the workbook is edited inside its zip (sheet XML), not through a spreadsheet library, so macros, buttons, drawings,
    comments, data validation and every other sheet stay byte for byte as they were;
  * the text of a cell is the `cell_name` of the assignment row in FlexibleSchedule.yaml (written by 3-2: the text the bar
    had in SU_Others, or a name made from fab and customer for a new module); only when a row has no cell_name the text of
    a real cell of the module is copied (the same worker's nearest cell), or, for a module with no cell in the sheet,
    the name is made here from the fab and customer (--new-module-text).  The style (colour, border, font) is the one of
    the nearest real cell of the module, or the most common module style of the sheet for a new module;
  * a cell that is no longer part of a module is emptied (the normal empty-cell style of the sheet);
  * cells of other tasks, red / grey cells and text of other modules are never overwritten;
  * a NEW module (added in step 2-4, so it has no cell in the sheet yet) is written with a name made from its fab and
    customer (--new-module-text) and the style of the most common module bar of the sheet (same colour theme as the other bars);
  * the FI / FO of the flexible trips are written too: FI on the first day of the trip, FO on the last day; where one trip ends
    and the next one starts on the same date only one mark (FO) is written, which is how the sheet is read back.

Inputs
  --schedule      schedule of the flexible modules (a module = a workflow task whose name holds the module code)
  --env           EnvConfig.yaml that belongs to it (worker id -> name)
  --su-others     the old SU_Others .xlsm (it is only read; give a copy, the output is a new file)
  --flexible-work optional <date>_FlexibleWork.yaml: the flexible trips of last week are taken from it (their FI/FO are removed
                  before the new ones are written).  It does NOT limit the modules: every module of the schedule is written,
                  including the modules that are new this week and so are not in the list.
Output (in --out-dir)
  <old file name>_rewritten.xlsm   the new workbook
  <old file name>_rewritten.txt    only with --report: every change, conflict and skipped item
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2-2-3prepare_schedule"))
import prepare_schedule as ps   # noqa: E402  (name_key, tool_code, ymd, load_yaml, import_decoder, read_prev_flexible)

DEFAULT_BLANK_STYLE = "4"       # style of an empty day cell in the real sheet; the tool uses the most common one it finds


# ----------------------------------------------------------------------------- column helpers

def col_index(letters):
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def col_letters(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ----------------------------------------------------------------------------- sheet xml

ROW_RE = re.compile(r'<row r="(\d+)"[^>]*?(?:/>|>.*?</row>)', re.S)
CELL_RE = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', re.S)


def parse_row_cells(row_xml):
    """[(col_index, attrs_without_r, inner)] of one <row>."""
    out = []
    for m in CELL_RE.finditer(row_xml):
        out.append((col_index(m.group(1)), m.group(3), m.group(4) or ""))
    return out


def cell_style(attrs):
    m = re.search(r'\bs="(\d+)"', attrs)
    return m.group(1) if m else None


def cell_xml(row, col, attrs, inner):
    ref = f"{col_letters(col)}{row}"
    return f'<c r="{ref}"{attrs}/>' if not inner else f'<c r="{ref}"{attrs}>{inner}</c>'


def index_sheet(xml):
    """{row: (start, end)} of every <row> inside <sheetData> and the {(row, col): (style, attrs, inner)} map of all cells."""
    rows, cells = {}, {}
    body_start = xml.index("<sheetData>")
    for m in ROW_RE.finditer(xml, body_start):
        r = int(m.group(1))
        rows[r] = (m.start(), m.end())
        for c, attrs, inner in parse_row_cells(m.group(0)):
            cells[(r, c)] = (cell_style(attrs), attrs, inner)
    return rows, cells


def patch_sheet_xml(xml, rows, new_cells):
    """
    new_cells: {(row, col): (attrs, inner)}  replaces the cell, or inserts it (in column order) when it does not exist.
    Everything outside the touched cells is copied unchanged.  Returns the new xml.
    """
    by_row = defaultdict(dict)
    for (r, c), v in new_cells.items():
        by_row[r][c] = v
    pieces, pos = [], 0
    for r in sorted(by_row):
        start, end = rows[r]
        row_xml = xml[start:end]
        if row_xml.endswith("/>") and "</row>" not in row_xml:      # <row .../> without cells
            head = row_xml[:-2] + ">"
            new = head + "".join(cell_xml(r, c, *by_row[r][c]) for c in sorted(by_row[r])) + "</row>"
        else:
            head_end = row_xml.index(">") + 1
            head, body = row_xml[:head_end], row_xml[head_end:-len("</row>")]
            out, last, pending = [], 0, sorted(by_row[r])
            for m in CELL_RE.finditer(body):
                c = col_index(m.group(1))
                while pending and pending[0] < c:          # a cell that does not exist yet goes before the next one
                    out.append(body[last:m.start()])
                    last = m.start()
                    k = pending.pop(0)
                    out.append(cell_xml(r, k, *by_row[r][k]))
                if pending and pending[0] == c:            # an existing cell is replaced
                    out.append(body[last:m.start()])
                    last = m.end()
                    out.append(cell_xml(r, c, *by_row[r][c]))
                    pending.pop(0)
            out.append(body[last:])
            out.extend(cell_xml(r, k, *by_row[r][k]) for k in pending)
            new = head + "".join(out) + "</row>"
        pieces.append(xml[pos:start])
        pieces.append(new)
        pos = end
    pieces.append(xml[pos:])
    return "".join(pieces)


# ----------------------------------------------------------------------------- the plan (pure)

def plan_changes(old_cells, want, rows_by_worker, col_by_date, empty, blocked, from_date, new_modules=None, names=None,
                 text_of=None):
    """
    old_cells : {code: {(row, col)}}   cells of the sheet that show the module now (red/grey cells excluded)
    want      : {code: {(worker_key, 'YYYY/MM/DD')}}   what the flexible schedule says
    rows_by_worker : {worker_key: [row]}   sheet rows of a worker
    col_by_date    : {'YYYY/MM/DD': col}   date columns of the sheet
    empty(row, col) -> True when the cell has no text;  blocked(row, col) -> True for red / grey cells
    from_date : cells before this date are never touched
    new_modules : {code: (text, style)} for modules that have no cell in the sheet yet (new modules): their cells are
                  written with this text and style; without an entry such a module is only reported
    names       : {code: {(worker_key, date): cell_name}} the text of the cells from the assignment rows (FlexibleSchedule)
    text_of(row, col) -> text of a sheet cell (or None); lets a cell whose real donor already has the cell_name be copied as it is
    Returns (clears [(code, row, col)], sets [(code, row, col, donor)], report [str]); donor is
      (row, col)                        a real cell of the module: its text and style are copied
      ("named", text, (row, col))       the style of that real cell, the text of the cell_name
      ("new", text, style)              a module with no cell in the sheet: the cell_name (or the made-up name) and the common style
    """
    report, clears, wants, names_at = [], [], {}, {}
    date_of_col = {c: d for d, c in col_by_date.items()}
    for code in sorted(want):
        old = old_cells.get(code, set())
        old_by_row = defaultdict(set)
        for r, c in old:
            old_by_row[r].add(c)
        target = set()
        name_at = names_at.setdefault(code, {})
        missing_worker = skipped_date = 0
        for wk, d in sorted(want[code]):
            if d < from_date:
                continue
            if wk not in rows_by_worker:
                missing_worker += 1
                continue
            col = col_by_date.get(d)
            if col is None:
                skipped_date += 1
                continue
            rows = rows_by_worker[wk]
            row = max(rows, key=lambda r: len(old_by_row.get(r, ())))    # the row that already carries the module, else the first
            target.add((row, col))
            nm = (names or {}).get(code, {}).get((wk, d))
            if nm:
                name_at[(row, col)] = nm
        wants[code] = target
        if missing_worker:
            report.append(f"WARN: module {code}: {missing_worker} worker-days skipped, the worker has no row in the sheet")
        if skipped_date:
            report.append(f"WARN: module {code}: {skipped_date} worker-days skipped, the date is outside the sheet")
        for r, c in sorted(old - target):
            if date_of_col.get(c, "0000/00/00") >= from_date:
                clears.append((code, r, c))

    freed = {(r, c) for _code, r, c in clears}
    claimed, sets, no_donor = {}, [], Counter()
    for code in sorted(wants):
        old = old_cells.get(code, set())
        for r, c in sorted(wants[code] - old):
            if blocked(r, c):
                report.append(f"CONFLICT: {code} row {r} {col_letters(c)}: the cell is red / grey, left as it is")
                continue
            if not (empty(r, c) or (r, c) in freed):
                report.append(f"CONFLICT: {code} row {r} {col_letters(c)}: the cell holds another task, left as it is")
                continue
            if (r, c) in claimed:
                report.append(f"CONFLICT: {code} row {r} {col_letters(c)}: already given to module {claimed[(r, c)]}")
                continue
            name = names_at.get(code, {}).get((r, c))
            if not old:
                if code not in (new_modules or {}):
                    no_donor[code] += 1
                    continue
                donor = ("new", name or new_modules[code][0], new_modules[code][1])
            else:
                donor = min(old, key=lambda rc: (rc[0] != r, abs(rc[1] - c), rc[0]))   # same worker row first, then nearest day
                if name and ps.cell_name_key(name) != ps.cell_name_key((text_of(*donor) if text_of else None) or ""):
                    donor = ("named", name, donor)       # the style of the real cell, the text of the cell_name
            claimed[(r, c)] = code
            sets.append((code, r, c, donor))
    for code, n in sorted(no_donor.items()):
        report.append(f"WARN: module {code} has no cell in the sheet, so its name and colour are unknown: {n} worker-days not written")
    for code in sorted({c for c, _r, _k, dn in sets if dn[0] == "new"}):
        texts = Counter(dn[1] for c, _r, _k, dn in sets if c == code and dn[0] == "new")
        report.append(f"NEW module {code}: {sum(texts.values())} worker-days written as "
                      + ", ".join(repr(t) for t in sorted(texts)) + f" (style {new_modules[code][1]})")
    return clears, sets, report


def plan_trips(want_trips, remove_trips, rows_by_worker, col_by_date, marks, texts_by_row, blocked, from_date):
    """
    FI / FO of the flexible trips.
    want_trips  : {worker_key: [(start, end)]}   flexible trips of the schedule (FI on start, FO on end)
    remove_trips: {worker_key: [(start, end)]}   flexible trips of last week (FlexibleWork): their FI / FO are removed first
    marks       : {(row, col): 'FI'|'FO'}        FI / FO cells of the sheet now
    texts_by_row: {row: {date: text}}            every written day of a row (to repair the pairs of the sheet)
    blocked(row, col) -> True for red / grey cells (never touched)
    A date where one trip ends and the next one starts holds ONE mark, FO (the FI is the same date, as the sheet is read back).
    Returns (clears [(row, col)], sets [(row, col, 'FI'|'FO')], report [str]).
    """
    report, clears, sets = [], [], []
    date_of_col = {c: d for d, c in col_by_date.items()}
    window = (min(col_by_date), max(col_by_date)) if col_by_date else (None, None)
    for wk in sorted(set(want_trips) | set(remove_trips)):
        rows = rows_by_worker.get(wk)
        if not rows:
            n = len(want_trips.get(wk, ())) + len(remove_trips.get(wk, ()))
            report.append(f"WARN: {n} trips of {wk} skipped, the worker has no row in the sheet")
            continue
        row = max(rows, key=lambda r: sum(1 for (rr, _c) in marks if rr == r))
        row_marks = {date_of_col[c]: m for (r, c), m in marks.items() if r == row and c in date_of_col}

        desired = {}
        trips = sorted(set(want_trips.get(wk, ())))
        for i, (s, e) in enumerate(trips):
            if s == e:
                report.append(f"WARN: one-day trip {wk} {s} cannot be written with one mark: skipped")
                continue
            touches_previous = any(e0 == s for (_s0, e0) in trips[:i])
            if not touches_previous:
                desired[s] = "FI"
            desired[e] = "FO"

        protected = set()
        if remove_trips.get(wk):
            pairs, _n = ps.pair_fifo_repaired(row_marks, list(texts_by_row.get(row, {})), window)
            old = set(remove_trips[wk])
            for s, e in pairs:
                if (s, e) not in old:
                    protected.update((s, e))
        freed = set()
        for s, e in sorted(set(remove_trips.get(wk, ()))):
            for d in (s, e):
                col = col_by_date.get(d)
                if col is None or d < from_date or d in desired or d in protected:
                    continue
                if (row, col) in marks and not blocked(row, col) and (row, col) not in freed:
                    clears.append((row, col))
                    freed.add((row, col))
        for d, mark in sorted(desired.items()):
            col = col_by_date.get(d)
            if col is None:
                report.append(f"WARN: {mark} of {wk} on {d} skipped, the date is outside the sheet")
                continue
            if d < from_date:
                continue
            if marks.get((row, col)) == mark:
                continue
            if blocked(row, col):
                report.append(f"CONFLICT: {mark} of {wk} on {d}: the cell is red / grey, left as it is")
                continue
            if (row, col) in marks and (row, col) not in freed:
                report.append(f"CONFLICT: {mark} of {wk} on {d}: the cell already holds {marks[(row, col)]}, left as it is")
                continue
            if d in texts_by_row.get(row, {}) and (row, col) not in marks:
                report.append(f"CONFLICT: {mark} of {wk} on {d}: the cell holds another task ({texts_by_row[row][d]!r}), left as it is")
                continue
            sets.append((row, col, mark))
    sets_set = {(r, c) for r, c, _m in sets}
    clears = [(r, c) for r, c in clears if (r, c) not in sets_set]       # a cleared cell that is written again is just replaced
    return clears, sets, report


# ----------------------------------------------------------------------------- reading the sheet (decoder7 helpers)

def sheet_part_name(zin, sheet_name):
    wb = zin.read("xl/workbook.xml").decode("utf-8")
    rels = zin.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    m = re.search(r'<sheet [^>]*?name="%s"[^>]*?r:id="(rId\d+)"' % re.escape(sheet_name), wb)
    if not m:
        names = re.findall(r'<sheet [^>]*?name="([^"]+)"', wb)
        raise ValueError(f"sheet {sheet_name!r} not found; available: {names}")
    t = re.search(r'<Relationship [^>]*?Id="%s"[^>]*?Target="([^"]+)"' % m.group(1), rels) or \
        re.search(r'<Relationship [^>]*?Target="([^"]+)"[^>]*?Id="%s"' % m.group(1), rels)
    return "xl/" + t.group(1).lstrip("/").replace("xl/", "", 1)


def read_layout(d, su_path, sheet_name):
    """Day cells, worker rows and module cells of the sheet, read the way decoder7 reads them."""
    from openpyxl import load_workbook
    wb = load_workbook(su_path, data_only=True, read_only=False)
    ws = wb[sheet_name]
    date_row_idx, date_cols, dt_by_col = d._find_date_header_ws(ws)
    cols = d._detect_su_columns(ws, date_row_idx + 1, stamp_row_idx=date_row_idx)
    col_by_date = {ps.ymd(dt_by_col[c]): c for c in date_cols}
    rows_by_worker, old_cells, texts, red_grey = defaultdict(list), defaultdict(set), {}, set()
    marks, texts_by_row = {}, defaultdict(dict)
    for r in range(date_row_idx + 2, ws.max_row + 1):
        name = ws.cell(r, cols["name"] + 1).value
        company = ws.cell(r, cols["company"] + 1).value
        if name is None or not str(name).strip() or not str(company or "").strip():
            continue
        rows_by_worker[ps.name_key(name)].append(r)
        for c in date_cols:
            cell = ws.cell(r, c)
            v = cell.value
            has_value = v is not None and str(v).strip() != ""
            if has_value:
                texts[(r, c)] = str(v)          # numbers count as "something written" too: never overwritten
                texts_by_row[r][ps.ymd(dt_by_col[c])] = str(v)
                if isinstance(v, str) and ps.read_mark(v):
                    marks[(r, c)] = ps.read_mark(v)
            # decoder7: a red cell is an unavailable day whatever it says; a grey cell is personal business unless it
            # carries a module code, then it is a module cell like any other
            if d._is_red_cell(cell):
                red_grey.add((r, c))
            elif has_value and isinstance(v, str) and d.extract_tool_code(d._normalize_tool_code_in_text(v.strip())):
                old_cells[d.extract_tool_code(d._normalize_tool_code_in_text(v.strip()))].add((r, c))
            elif d._is_grey_cell(cell) and not has_value:
                red_grey.add((r, c))
    return {"col_by_date": col_by_date, "rows_by_worker": dict(rows_by_worker), "old_cells": dict(old_cells),
            "texts": texts, "red_grey": red_grey, "date_cols": sorted(date_cols), "marks": marks,
            "texts_by_row": dict(texts_by_row)}


# ----------------------------------------------------------------------------- write

def inline_string_cell(text, style):
    """(attrs, inner) of a cell with an inline string: no change of sharedStrings.xml is needed for a new text."""
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    space = ' xml:space="preserve"' if esc != esc.strip() else ""
    return f' s="{style}" t="inlineStr"', f"<is><t{space}>{esc}</t></is>"


def rewrite(su_path, out_path, sheet_name, layout, clears, sets, sheet_xml, rows, cells, blank_style):
    """Writes the new workbook: every zip member copied unchanged except the sheet and the shared-string counter.
    clears: [(code, row, col)];  sets: [(code, row, col, donor)] with donor = (row, col) of a real cell, ("named", text, (row, col))
    or ("new", text, style)."""
    new_cells, d_text = {}, 0
    for _code, r, c in clears:
        old_inner = cells.get((r, c), (None, "", ""))
        d_text -= 1 if 't="s"' in old_inner[1] else 0
        new_cells[(r, c)] = (f' s="{blank_style}"', "")
    for _code, r, c, donor in sets:
        if donor[0] == "new":
            attrs, inner = inline_string_cell(donor[1], donor[2])
        elif donor[0] == "named":
            attrs, inner = inline_string_cell(donor[1], cells[donor[2]][0])
        else:
            _s, attrs, inner = cells[donor]
        old_attrs = cells.get((r, c), (None, "", ""))[1]
        d_text += (1 if 't="s"' in attrs else 0) - (1 if 't="s"' in old_attrs else 0)
        new_cells[(r, c)] = (attrs, inner)
    new_xml = patch_sheet_xml(sheet_xml, rows, new_cells)
    with zipfile.ZipFile(su_path) as zin, zipfile.ZipFile(out_path, "w") as zout:
        part = sheet_part_name(zin, sheet_name)
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == part:
                data = new_xml.encode("utf-8")
            elif info.filename == "xl/sharedStrings.xml" and d_text:
                s = data.decode("utf-8")
                s = re.sub(r'(<sst [^>]*?\bcount=")(\d+)(")', lambda m: f"{m.group(1)}{int(m.group(2)) + d_text}{m.group(3)}", s, count=1)
                data = s.encode("utf-8")
            zout.writestr(info, data, compress_type=info.compress_type)
    return new_cells


def verify(su_path, out_path, sheet_name, new_cells):
    """XML level check: only the sheet (and the shared-string counter) differ, and only the planned cells changed."""
    problems = []
    with zipfile.ZipFile(su_path) as a, zipfile.ZipFile(out_path) as b:
        if [i.filename for i in a.infolist()] != [i.filename for i in b.infolist()]:
            problems.append("the list of zip members differs")
        part = sheet_part_name(a, sheet_name)
        for info in a.infolist():
            if info.filename in (part, "xl/sharedStrings.xml"):
                continue
            if hashlib.sha1(a.read(info.filename)).digest() != hashlib.sha1(b.read(info.filename)).digest():
                problems.append(f"{info.filename} changed")
        xa, xb = a.read(part).decode("utf-8"), b.read(part).decode("utf-8")
        _ra, ca = index_sheet(xa)
        _rb, cb = index_sheet(xb)
        diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        if diff - set(new_cells):
            problems.append(f"{len(diff - set(new_cells))} cells changed that were not planned")
        for k, (attrs, inner) in new_cells.items():
            if k not in cb or cb[k][1] != attrs or cb[k][2] != inner:
                problems.append(f"cell {col_letters(k[1])}{k[0]} was not written as planned")
        head_a, head_b = xa[:xa.index("<sheetData>")], xb[:xb.index("<sheetData>")]
        tail_a, tail_b = xa[xa.index("</sheetData>"):], xb[xb.index("</sheetData>"):]
        if head_a != head_b or tail_a != tail_b:
            problems.append("the sheet outside <sheetData> changed")
    return problems


# ----------------------------------------------------------------------------- main

def wanted_cells(schedule_doc, env_doc, only_modules=None):
    """({code: {(worker_key, date)}}, warnings, {code: {"fab", "customer", "description"}}, {worker_key: [(start, end)]},
    {code: {(worker_key, date): cell_name}}).
    Every module with assignments is wanted (only_modules, if given, limits it); trips are the Flexible ones of the schedule."""
    env = env_doc.get("environment", env_doc)
    names = {w["id"]: w["name"] for w in env.get("worker_list") or []}
    prev = ps.read_prev_flexible(schedule_doc, names)
    want, info, cell_names = {}, {}, {}
    for code, m in prev["modules"].items():
        if not m["assign"]:
            continue                      # a module without any assignment is not rewritten (nothing says where it is)
        if only_modules is not None and code not in only_modules:
            continue
        want[code] = set(m["assign"])
        info[code] = ps.module_name_info(m["task"], env)
        cell_names[code] = dict(m["names"])
    trips = defaultdict(list)
    for t in prev["trips"]:
        if t["flex"].lower() == "flexible":
            trips[t["worker_key"]].append((t["start"], t["end"]))
    return want, prev["warnings"], info, dict(trips), cell_names


def new_module_text(template, code, info):
    """Name written into the cells of a new module, e.g. '{customer}{fab}_{code}_新規' -> 'CustAFabA_530N00000A_新規'."""
    return ps.module_cell_name(code, info, template)


def most_common_module_style(old_cells, cells, style_from=None):
    """Style id of the module bars: the style of module `style_from`, or the most common one among all module cells."""
    pool = old_cells.get(style_from) if style_from else None
    if style_from and not pool:
        raise ValueError(f"--new-module-style-from {style_from}: that module has no cell in the sheet")
    pool = pool or {rc for cs in old_cells.values() for rc in cs}
    styles = Counter(cells[rc][0] for rc in pool if rc in cells and cells[rc][0])
    return styles.most_common(1)[0][0] if styles else DEFAULT_BLANK_STYLE


def mark_donors(layout, cells):
    """{'FI': (row, col), 'FO': (row, col)}: a real FI / FO cell with the most common style (text exactly FI / FO)."""
    out = {}
    for mark in ("FI", "FO"):
        pool = [rc for rc, t in layout["texts"].items() if t.strip().upper() == mark and rc in cells and rc not in layout["red_grey"]]
        if not pool:
            continue
        top = Counter(cells[rc][0] for rc in pool).most_common(1)[0][0]
        out[mark] = next(rc for rc in sorted(pool) if cells[rc][0] == top)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 3-3: write the flexible schedule back into a copy of SU_Others.")
    ap.add_argument("--schedule", required=True, help="schedule of the flexible modules (FlexibleSchedule / decoder7 / Timefold result Schedule.yaml)")
    ap.add_argument("--env", required=True, help="EnvConfig.yaml that belongs to the schedule (worker names)")
    ap.add_argument("--su-others", required=True, help="old SU_Others workbook (.xlsm) (only read)")
    ap.add_argument("--out-dir", required=True, help="folder for the new workbook")
    ap.add_argument("--flexible-work", default=None,
                    help="<date>_FlexibleWork.yaml: its flexible trips are removed from the sheet before the new FI/FO are written "
                         "(it does not limit the modules)")
    ap.add_argument("--new-module-text", default=ps.DEFAULT_CELL_NAME_TEMPLATE,
                    help="text of the cells of a module that has no cell in the sheet and no cell_name in the schedule rows; "
                         "{customer} {fab} {code} {description} (default '{customer}{fab}_{code}_新規')")
    ap.add_argument("--new-module-style-from", default=None,
                    help="module code whose cell style is used for new modules (default: the most common module style of the sheet)")
    ap.add_argument("--no-trips", action="store_true", help="do not write FI/FO")
    ap.add_argument("--sheet", default="予定表_2026", help="task sheet of SU_Others")
    ap.add_argument("--from-date", default=None, help="YYYY/MM/DD; cells before it are never touched (default: today)")
    ap.add_argument("--report", action="store_true", help="also write <name>_rewritten.txt with every change and conflict")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan, write nothing")
    ap.add_argument("--decoder-dir", default=str(next((d for d in (HERE.parents[1] / "decoder", HERE.parents[1] / "decorder") if d.exists()), HERE.parents[1] / "decoder")), help="folder of decoder7.py")
    args = ap.parse_args(argv)

    src = Path(args.su_others)
    out = Path(args.out_dir) / f"{src.stem}_rewritten{src.suffix}"
    if out.resolve() == src.resolve():
        ap.error("the output would overwrite the input")
    from_date = ps.ymd(args.from_date) if args.from_date else date.today().strftime("%Y/%m/%d")
    if from_date is None:
        ap.error("--from-date must look like 2026/10/01")

    env_doc = ps.load_yaml(args.env)
    want_names = [w["name"] for w in env_doc.get("environment", env_doc).get("worker_list") or []]
    want, warns, info, want_trips, cell_names = wanted_cells(ps.load_yaml(args.schedule), env_doc)
    remove_trips, listed_modules = {}, set()
    if args.flexible_work:
        names = {w["id"]: w["name"] for w in (env_doc.get("environment", env_doc).get("worker_list") or [])}
        fw = ps.load_yaml(args.flexible_work).get("flexible_work") or {}
        listed_modules = set(fw.get("modules") or [])
        for t in fw.get("trips") or []:
            s, e = ps.ymd(t.get("start_date")), ps.ymd(t.get("end_date"))
            if s and e and t.get("worker") in names:
                remove_trips.setdefault(ps.name_key(names[t["worker"]]), []).append((s, e))
    print(f"{len(want)} modules with assignments and {sum(len(v) for v in want_trips.values())} flexible trips in the schedule; "
          f"reading SU_Others (this takes a while) ...", flush=True)
    d = ps.import_decoder(args.decoder_dir)
    layout = read_layout(d, str(src), args.sheet)

    with zipfile.ZipFile(src) as z:
        part = sheet_part_name(z, args.sheet)
        sheet_xml = z.read(part).decode("utf-8")
    rows, cells = index_sheet(sheet_xml)
    blank = Counter(cells[(r, c)][0] for rs in layout["rows_by_worker"].values() for r in rs for c in layout["date_cols"]
                    if (r, c) in cells and (r, c) not in layout["texts"] and (r, c) not in layout["red_grey"])
    blank_style = blank.most_common(1)[0][0] if blank else DEFAULT_BLANK_STYLE

    new_style = most_common_module_style(layout["old_cells"], cells, args.new_module_style_from)
    new_modules = {code: (new_module_text(args.new_module_text, code, info.get(code, {})), new_style)
                   for code in want if not layout["old_cells"].get(code)}
    clears, sets, report = plan_changes(
        layout["old_cells"], want, layout["rows_by_worker"], layout["col_by_date"],
        empty=lambda r, c: (r, c) not in layout["texts"], blocked=lambda r, c: (r, c) in layout["red_grey"], from_date=from_date,
        new_modules=new_modules, names=cell_names, text_of=lambda r, c: layout["texts"].get((r, c)))
    no_name = sorted(code for code in want if not cell_names.get(code))
    if no_name:
        report.append(f"INFO: no cell_name in the schedule rows of {', '.join(no_name)}: the text of a real cell of the module is "
                      f"copied (or the name is made from fab and customer for a new module)")
    report = [f"WARN: {w}" for w in warns] + report
    if listed_modules:
        for code in sorted(listed_modules - set(want)):
            report.append(f"WARN: module {code} is in FlexibleWork but has no assignment in the schedule: its cells are left as they are")
        for code in sorted(set(want) - listed_modules):
            report.append(f"INFO: module {code} is in the schedule but not in FlexibleWork (new this week)")
    trip_clears, trip_sets, trip_report = [], [], []
    if not args.no_trips and (want_trips or remove_trips):
        donors = mark_donors(layout, cells)
        if len(donors) < 2:
            trip_report.append("WARN: the sheet has no plain FI / FO cell to copy the style from: FI / FO not written")
        else:
            trip_clears, trip_sets, trip_report = plan_trips(
                want_trips, remove_trips, layout["rows_by_worker"], layout["col_by_date"], layout["marks"],
                layout["texts_by_row"], lambda r, c: (r, c) in layout["red_grey"], from_date)
    report += trip_report
    clears = clears + [("FI/FO", r, c) for r, c in trip_clears]
    sets = sets + [("FI/FO", r, c, donors[m]) for r, c, m in trip_sets] if trip_sets else sets
    per = defaultdict(lambda: [0, 0])
    for code, _r, _c in clears:
        per[code][0] += 1
    for code, _r, _c, _dn in sets:
        per[code][1] += 1
    per_trip = per.pop("FI/FO", [0, 0])
    lines = [f"write_su_others report (from {from_date}, sheet {args.sheet})",
             f"{len(clears) - per_trip[0]} module cells emptied, {len(sets) - per_trip[1]} module cells filled, "
             f"{sum(1 for r in report if r.startswith('CONFLICT'))} conflicts "
             f"(FI/FO: {per_trip[0]} emptied, {per_trip[1]} written)", ""]
    lines += [f"module {code}: {a} cells emptied, {b} cells filled" for code, (a, b) in sorted(per.items())]
    lines += [""] + report
    # every cell that is written or emptied, with the worker and the date, to find the changes in Excel (Ctrl+G, cell address)
    name_of_row = {r: wk for wk, rs in layout["rows_by_worker"].items() for r in rs}
    date_of_col = {c: dt for dt, c in layout["col_by_date"].items()}
    display = {ps.name_key(n): n for n in want_names}
    lines += ["", "CELLS (cell, worker, date, what)"]
    for code, r, c in clears:
        lines.append(f"EMPTIED {col_letters(c)}{r}\t{display.get(name_of_row.get(r), name_of_row.get(r))}\t{date_of_col.get(c)}\t{code}")
    for code, r, c, dn in sets:
        what = dn[1] if dn[0] == "new" else layout["texts"].get(dn, "")
        lines.append(f"WRITTEN {col_letters(c)}{r}\t{display.get(name_of_row.get(r), name_of_row.get(r))}\t{date_of_col.get(c)}\t{what}")
    print("\n".join(lines[:2]))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    new_cells = rewrite(str(src), str(out), args.sheet, layout, clears, sets, sheet_xml, rows, cells, blank_style)
    problems = verify(str(src), str(out), args.sheet, new_cells)
    if problems:
        print("VERIFY FAILED: " + "; ".join(problems[:5]), file=sys.stderr)
        return 3
    print(f"written {out}")
    print("check: only the sheet cells listed in the plan differ; all other parts of the file are byte-identical")
    if args.report:
        rp = out.with_suffix(".txt")
        rp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
