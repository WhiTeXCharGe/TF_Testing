import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (common)
import write_su_others as w   # noqa: E402
from common.text import name_key   # noqa: E402

SHEET = (
    '<?xml version="1.0"?><worksheet><sheetPr/><dimension ref="A1:E3"/><sheetData>'
    '<row r="1"><c r="A1" s="1"/></row>'
    '<row r="2" spans="1:5"><c r="B2" s="9" t="s"><v>7</v></c><c r="C2" s="9" t="s"><v>7</v></c><c r="E2" s="4"/></row>'
    '<row r="3"/>'
    '</sheetData><autoFilter ref="A1:E3"/></worksheet>')


class PatchXml(unittest.TestCase):
    def setUp(self):
        self.rows, self.cells = w.index_sheet(SHEET)

    def test_index_finds_rows_and_cells(self):
        self.assertEqual({1, 2, 3}, set(self.rows))
        self.assertEqual(("9", ' s="9" t="s"', "<v>7</v>"), self.cells[(2, 2)])
        self.assertEqual(("4", ' s="4"', ""), self.cells[(2, 5)])

    def test_replace_insert_before_between_after_and_empty_row(self):
        new = {(2, 3): (' s="4"', ""),                        # replace C2 by an empty cell
               (2, 1): (' s="9" t="s"', "<v>7</v>"),           # A2 does not exist: goes first
               (2, 4): (' s="9" t="s"', "<v>7</v>"),           # D2 between C2 and E2
               (2, 6): (' s="9" t="s"', "<v>7</v>"),           # F2 after the last cell
               (3, 2): (' s="9" t="s"', "<v>7</v>")}           # row 3 had no cells
        xml = w.patch_sheet_xml(SHEET, self.rows, new)
        _r, cells = w.index_sheet(xml)
        self.assertEqual(["A2", "B2", "C2", "D2", "E2", "F2"],
                         [w.col_letters(c) + "2" for (r, c) in sorted(cells) if r == 2])
        self.assertEqual(("4", ' s="4"', ""), cells[(2, 3)])
        self.assertEqual(("9", ' s="9" t="s"', "<v>7</v>"), cells[(3, 2)])
        self.assertEqual(("9", ' s="9" t="s"', "<v>7</v>"), cells[(2, 2)])    # B2 untouched
        self.assertTrue(xml.startswith(SHEET[:SHEET.index('<row r="2"')]))    # everything before the first touched row is unchanged
        self.assertTrue(xml.endswith('</sheetData><autoFilter ref="A1:E3"/></worksheet>'))
        self.assertEqual(('<row r="1"><c r="A1" s="1"/></row>' in xml), True)

    def test_column_letters_round_trip(self):
        for n in (1, 26, 27, 52, 53, 469, 701, 702, 703):
            self.assertEqual(n, w.col_index(w.col_letters(n)))
        self.assertEqual("RW", w.col_letters(w.col_index("RW")))


def days(*d):
    return {x: i + 10 for i, x in enumerate(d)}


class Plan(unittest.TestCase):
    COLS = {f"2026/12/{n:02d}": 9 + n for n in range(1, 21)}       # 12/01 -> col 10 ... 12/20 -> col 29
    ROWS = {"yamada": [5], "sato": [6], "twin": [7, 8]}

    def plan(self, old, want, texts=(), red=(), from_date="2026/12/01"):
        return w.plan_changes(old, want, self.ROWS, self.COLS,
                              empty=lambda r, c: (r, c) not in set(texts), blocked=lambda r, c: (r, c) in set(red),
                              from_date=from_date)

    def test_nothing_to_do_when_the_sheet_already_matches(self):
        old = {"M1": {(5, 10), (5, 11)}}
        clears, sets, report = self.plan(old, {"M1": {("yamada", "2026/12/01"), ("yamada", "2026/12/02")}}, texts=old["M1"])
        self.assertEqual(([], [], []), (clears, sets, report))

    def test_a_shifted_module_empties_old_cells_and_fills_new_ones_from_the_same_row(self):
        old = {"M1": {(5, 10), (5, 11)}}
        want = {"M1": {("yamada", "2026/12/03"), ("yamada", "2026/12/04")}}
        clears, sets, _ = self.plan(old, want, texts=old["M1"])
        self.assertEqual({("M1", 5, 10), ("M1", 5, 11)}, set(clears))
        self.assertEqual({("M1", 5, 12, (5, 11)), ("M1", 5, 13, (5, 11))}, set(sets))     # donor = nearest cell of that row

    def test_another_worker_takes_the_style_of_the_nearest_cell_of_the_module(self):
        old = {"M1": {(5, 10), (5, 20)}}
        _c, sets, _ = self.plan(old, {"M1": {("yamada", "2026/12/01"), ("yamada", "2026/12/11"), ("sato", "2026/12/09")}},
                                texts=old["M1"])
        self.assertEqual([("M1", 6, 18, (5, 20))], [s for s in sets if s[1] == 6])

    def test_cells_before_from_date_are_never_touched(self):
        old = {"M1": {(5, 10), (5, 12)}}
        clears, sets, _ = self.plan(old, {"M1": {("yamada", "2026/12/05")}}, texts=old["M1"], from_date="2026/12/03")
        self.assertEqual({("M1", 5, 12)}, set(clears))        # 12/01 (col 10) is in the past: kept
        self.assertEqual([("M1", 5, 14, (5, 12))], [(c, r, k, d) for c, r, k, d in sets])

    def test_cells_with_other_tasks_and_red_cells_are_conflicts_not_overwritten(self):
        old = {"M1": {(5, 10)}}
        texts = {(5, 10), (5, 11), (5, 12)}
        clears, sets, report = self.plan(old, {"M1": {("yamada", "2026/12/02"), ("yamada", "2026/12/03"), ("yamada", "2026/12/04")}},
                                         texts=texts, red={(5, 13)})
        self.assertEqual([], sets)
        self.assertEqual(3, sum(1 for r in report if r.startswith("CONFLICT")))

    def test_a_cell_freed_by_one_module_can_be_taken_by_another(self):
        old = {"M1": {(5, 10)}, "M2": {(6, 10)}}
        want = {"M1": {("yamada", "2026/12/02")}, "M2": {("sato", "2026/12/01"), ("yamada", "2026/12/01")}}
        clears, sets, report = self.plan(old, want, texts={(5, 10), (6, 10)})
        self.assertIn(("M2", 5, 10, (6, 10)), sets)           # (5,10) was M1's and is free now
        self.assertEqual([], [r for r in report if r.startswith("CONFLICT")])

    def test_two_modules_cannot_take_the_same_cell(self):
        old = {"M1": {(5, 20)}, "M2": {(6, 20)}}
        want = {"M1": {("yamada", "2026/12/03")}, "M2": {("yamada", "2026/12/03")}}
        _c, sets, report = self.plan(old, want, texts={(5, 20), (6, 20)})
        self.assertEqual(1, len(sets))
        self.assertTrue(any("already given to module M1" in r for r in report))

    def test_unknown_worker_date_outside_sheet_and_module_without_cells_are_reported(self):
        old = {"M1": {(5, 10)}}
        want = {"M1": {("nobody", "2026/12/02"), ("yamada", "2027/05/01")}, "M9": {("sato", "2026/12/02")}}
        clears, sets, report = self.plan(old, want, texts={(5, 10)})
        text = "\n".join(report)
        self.assertIn("M1: 1 worker-days skipped, the worker has no row", text)
        self.assertIn("M1: 1 worker-days skipped, the date is outside the sheet", text)
        self.assertIn("module M9 has no cell in the sheet", text)
        self.assertEqual([("M1", 5, 10)], clears)              # nothing placeable: the old cell is emptied
        self.assertEqual([], sets)

    def test_a_worker_with_two_rows_uses_the_row_that_already_carries_the_module(self):
        old = {"M1": {(8, 10)}}
        _c, sets, _ = self.plan(old, {"M1": {("twin", "2026/12/01"), ("twin", "2026/12/02")}}, texts=old["M1"])
        self.assertEqual([("M1", 8, 11, (8, 10))], sets)


class Zip(unittest.TestCase):
    def build(self, path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr("xl/workbook.xml", '<workbook><sheets><sheet name="予定表_2026" sheetId="3" r:id="rId3"/></sheets></workbook>')
            z.writestr("xl/_rels/workbook.xml.rels",
                       '<Relationships><Relationship Id="rId3" Type="x" Target="worksheets/sheet3.xml"/></Relationships>')
            z.writestr("xl/worksheets/sheet3.xml", SHEET)
            z.writestr("xl/sharedStrings.xml", '<sst xmlns="x" count="2" uniqueCount="1"><si><t>M1 text</t></si></sst>')
            z.writestr("xl/vbaProject.bin", b"\x00\x01binary")

    def test_rewrite_changes_only_the_planned_cells_and_verify_passes(self):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "in.xlsm", Path(td) / "out.xlsm"
            self.build(src)
            rows, cells = w.index_sheet(SHEET)
            clears = [("M1", 2, 3)]                                  # empty C2
            sets = [("M1", 3, 4, (2, 2))]                            # D3 gets the style / text of B2
            new = w.rewrite(str(src), str(out), "予定表_2026", None, clears, sets, SHEET, rows, cells, "4")
            self.assertEqual([], w.verify(str(src), str(out), "予定表_2026", new))
            with zipfile.ZipFile(out) as z:
                _r, c2 = w.index_sheet(z.read("xl/worksheets/sheet3.xml").decode("utf-8"))
                self.assertEqual(("4", ' s="4"', ""), c2[(2, 3)])
                self.assertEqual(("9", ' s="9" t="s"', "<v>7</v>"), c2[(3, 4)])
                self.assertIn('count="2"', z.read("xl/sharedStrings.xml").decode("utf-8"))     # one text removed, one added
                self.assertEqual(b"\x00\x01binary", z.read("xl/vbaProject.bin"))

    def test_verify_detects_a_changed_member(self):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "in.xlsm", Path(td) / "out.xlsm"
            self.build(src)
            rows, cells = w.index_sheet(SHEET)
            new = w.rewrite(str(src), str(out), "予定表_2026", None, [("M1", 2, 3)], [], SHEET, rows, cells, "4")
            tampered = Path(td) / "t.xlsm"
            with zipfile.ZipFile(out) as zi, zipfile.ZipFile(tampered, "w") as zo:
                for info in zi.infolist():
                    zo.writestr(info, b"changed" if info.filename == "xl/vbaProject.bin" else zi.read(info.filename))
            self.assertTrue(any("vbaProject" in p for p in w.verify(str(src), str(tampered), "予定表_2026", new)))


class NewModules(unittest.TestCase):
    COLS = {f"2026/12/{n:02d}": 9 + n for n in range(1, 21)}
    ROWS = {"yamada": [5], "sato": [6]}

    def plan(self, old, want, new_modules, texts=()):
        return w.plan_changes(old, want, self.ROWS, self.COLS, empty=lambda r, c: (r, c) not in set(texts),
                              blocked=lambda r, c: False, from_date="2026/12/01", new_modules=new_modules)

    def test_a_module_without_any_cell_is_written_with_the_given_name_and_style(self):
        want = {"N1": {("yamada", "2026/12/03"), ("yamada", "2026/12/04"), ("sato", "2026/12/03")}}
        clears, sets, report = self.plan({}, want, {"N1": ("CustAFabA_N1_新規", "115")})
        self.assertEqual([], clears)
        self.assertEqual(3, len(sets))
        self.assertTrue(all(dn == ("new", "CustAFabA_N1_新規", "115") for _c, _r, _k, dn in sets))
        self.assertTrue(any(r.startswith("NEW module N1") for r in report))

    def test_without_a_new_module_entry_it_is_still_only_reported(self):
        _c, sets, report = self.plan({}, {"N1": {("yamada", "2026/12/03")}}, None)
        self.assertEqual([], sets)
        self.assertTrue(any("has no cell in the sheet" in r for r in report))

    def test_a_new_module_does_not_overwrite_a_cell_with_another_task(self):
        _c, sets, report = self.plan({}, {"N1": {("yamada", "2026/12/03")}}, {"N1": ("N1", "115")}, texts={(5, 12)})
        self.assertEqual([], sets)
        self.assertTrue(any(r.startswith("CONFLICT") for r in report))

    def test_text_template_and_styles(self):
        info = {"fab": "FabA", "customer": "CustA", "description": "x"}
        self.assertEqual("CustAFabA_530N00000A_新規", w.new_module_text("{customer}{fab}_{code}_新規", "530N00000A", info))
        self.assertEqual("530N00000A_新規", w.new_module_text("{customer}{fab}_{code}_新規", "530N00000A", {}))   # no fab / customer
        self.assertEqual("530N00000A", w.new_module_text("{fab}", "530N00000A", {}))                          # nothing left: the code
        cells = {(5, 10): ("115", "", ""), (5, 11): ("115", "", ""), (6, 10): ("3", "", "")}
        self.assertEqual("115", w.most_common_module_style({"A": {(5, 10), (5, 11)}, "B": {(6, 10)}}, cells))
        self.assertEqual("3", w.most_common_module_style({"A": {(5, 10), (5, 11)}, "B": {(6, 10)}}, cells, style_from="B"))

    def test_inline_string_cell_is_escaped_and_survives_rewrite_and_verify(self):
        attrs, inner = w.inline_string_cell("A&B <x>", "115")
        self.assertEqual(' s="115" t="inlineStr"', attrs)
        self.assertEqual("<is><t>A&amp;B &lt;x&gt;</t></is>", inner)
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "in.xlsm", Path(td) / "out.xlsm"
            Zip().build(src)
            rows, cells = w.index_sheet(SHEET)
            new = w.rewrite(str(src), str(out), "予定表_2026", None, [], [("N1", 3, 4, ("new", "CustA_N1_新規", "115"))], SHEET, rows, cells, "4")
            self.assertEqual([], w.verify(str(src), str(out), "予定表_2026", new))
            with zipfile.ZipFile(out) as z:
                _r, c2 = w.index_sheet(z.read("xl/worksheets/sheet3.xml").decode("utf-8"))
                self.assertEqual(("115", ' s="115" t="inlineStr"', "<is><t>CustA_N1_新規</t></is>"), c2[(3, 4)])
                self.assertIn('count="2"', z.read("xl/sharedStrings.xml").decode("utf-8"))      # no shared string touched


class CellNames(unittest.TestCase):
    """The text of a cell is the cell_name of the schedule row (written by 3-2); a real cell's text is only the fallback."""
    COLS = {f"2026/12/{n:02d}": 9 + n for n in range(1, 21)}
    ROWS = {"yamada": [5], "sato": [6]}
    TEXTS = {(5, 10): "OLD_M1", (5, 11): "OLD_M1"}

    def plan(self, old, want, names, new_modules=None):
        return w.plan_changes(old, want, self.ROWS, self.COLS, empty=lambda r, c: (r, c) not in self.TEXTS,
                              blocked=lambda r, c: False, from_date="2026/12/01", new_modules=new_modules, names=names,
                              text_of=lambda r, c: self.TEXTS.get((r, c)))

    def test_a_new_cell_gets_the_cell_name_with_the_style_of_the_nearest_real_cell(self):
        old = {"M1": {(5, 10), (5, 11)}}
        want = {"M1": {("yamada", "2026/12/01"), ("yamada", "2026/12/02"), ("yamada", "2026/12/03")}}
        names = {"M1": {("yamada", "2026/12/03"): "CELL_NAME_M1"}}
        _c, sets, _r = self.plan(old, want, names)
        self.assertEqual([("M1", 5, 12, ("named", "CELL_NAME_M1", (5, 11)))], sets)

    def test_when_the_real_cell_already_has_that_text_it_is_copied_as_it_is(self):
        old = {"M1": {(5, 10), (5, 11)}}
        want = {"M1": {("yamada", "2026/12/01"), ("yamada", "2026/12/02"), ("yamada", "2026/12/03")}}
        _c, sets, _r = self.plan(old, want, {"M1": {("yamada", "2026/12/03"): "old_ m1"}})        # same text for the comparison key
        self.assertEqual([("M1", 5, 12, (5, 11))], sets)

    def test_without_a_cell_name_the_real_cell_is_copied(self):
        old = {"M1": {(5, 10), (5, 11)}}
        want = {"M1": {("yamada", "2026/12/01"), ("yamada", "2026/12/02"), ("yamada", "2026/12/03")}}
        _c, sets, _r = self.plan(old, want, {"M1": {}})
        self.assertEqual([("M1", 5, 12, (5, 11))], sets)

    def test_a_new_module_uses_its_cell_name_and_falls_back_to_the_made_up_name(self):
        want = {"N1": {("yamada", "2026/12/03"), ("sato", "2026/12/03")}}
        names = {"N1": {("yamada", "2026/12/03"): "FROM_ROW"}}
        _c, sets, _r = self.plan({}, want, names, new_modules={"N1": ("MADE_UP_N1_新規", "115")})
        self.assertEqual({(5, 12): ("new", "FROM_ROW", "115"), (6, 12): ("new", "MADE_UP_N1_新規", "115")},
                         {(r, c): dn for _code, r, c, dn in sets})

    def test_named_cell_is_written_as_an_inline_string_with_the_style_of_the_real_cell(self):
        with tempfile.TemporaryDirectory() as td:
            src, out = Path(td) / "in.xlsm", Path(td) / "out.xlsm"
            Zip().build(src)
            rows, cells = w.index_sheet(SHEET)
            new = w.rewrite(str(src), str(out), "予定表_2026", None, [], [("M1", 3, 4, ("named", "NAME & CO", (2, 2)))], SHEET, rows, cells, "4")
            self.assertEqual([], w.verify(str(src), str(out), "予定表_2026", new))
            with zipfile.ZipFile(out) as z:
                _r, c2 = w.index_sheet(z.read("xl/worksheets/sheet3.xml").decode("utf-8"))
                self.assertEqual(("9", ' s="9" t="inlineStr"', "<is><t>NAME &amp; CO</t></is>"), c2[(3, 4)])

    def test_wanted_cells_reads_the_names_and_the_fab_customer_info(self):
        doc = {"schedule": {"workflow_task_list": [{"id": "e1", "name": "530N00001A", "fab": "f1", "phase_task_list": [
            {"id": "e1p1", "operation_task_list": [{"id": "e1p1o1"}]}]}],
            "assignment_list": [{"worker": "w1", "operation_task": "e1p1o1", "start_date": "2026/12/03", "end_date": "2026/12/03",
                                 "work_date_list": [{"date": "2026/12/03", "hour": 10}], "plan_flexibility": "Flexible",
                                 "cell_name": "THE_NAME"}]}}
        env = {"environment": {"worker_list": [{"id": "w1", "name": "山田 太郎"}],
                               "fab_list": [{"id": "f1", "name": "FabA", "customer_company": "c1"}],
                               "customer_company_list": [{"id": "c1", "name": "CustA"}]}}
        want, _warn, info, _trips, names = w.wanted_cells(doc, env)
        self.assertEqual({"530N00001A": {(name_key("山田 太郎"), "2026/12/03")}}, want)
        self.assertEqual({"530N00001A": {(name_key("山田 太郎"), "2026/12/03"): "THE_NAME"}}, names)
        self.assertEqual({"fab": "FabA", "customer": "CustA", "description": ""}, info["530N00001A"])


class Trips(unittest.TestCase):
    COLS = {f"2026/12/{n:02d}": 9 + n for n in range(1, 31)}
    ROWS = {"yamada": [5]}

    def plan(self, want, remove=None, marks=None, texts=None, blocked=()):
        marks = marks or {}
        by_row = {5: texts or {}}
        return w.plan_trips({"yamada": want}, {"yamada": remove} if remove else {}, self.ROWS, self.COLS, marks, by_row,
                            lambda r, c: (r, c) in set(blocked), "2026/12/01")

    def test_fi_on_the_first_day_and_fo_on_the_last_day(self):
        clears, sets, report = self.plan([("2026/12/05", "2026/12/10")])
        self.assertEqual([], clears)
        self.assertEqual({(5, 14, "FI"), (5, 19, "FO")}, set(sets))

    def test_trips_that_touch_share_one_fo_mark(self):
        clears, sets, _ = self.plan([("2026/12/05", "2026/12/10"), ("2026/12/10", "2026/12/18")])
        self.assertEqual({(5, 14, "FI"), (5, 19, "FO"), (5, 27, "FO")}, set(sets))     # no FI on 12/10

    def test_old_flexible_trip_marks_are_removed_unless_written_again(self):
        marks = {(5, 14): "FI", (5, 19): "FO"}
        clears, sets, _ = self.plan([("2026/12/05", "2026/12/12")], remove=[("2026/12/05", "2026/12/10")], marks=marks,
                                    texts={"2026/12/05": "FI", "2026/12/10": "FO"})
        self.assertEqual([(5, 19)], clears)                      # old FO on 12/10 goes, FI on 12/05 stays
        self.assertEqual([(5, 21, "FO")], sets)

    def test_a_mark_that_belongs_to_another_trip_of_the_sheet_is_protected(self):
        # sheet: flexible trip 12/05-12/10 (to remove) and a Fixed trip 12/10-12/18 that starts on the same date (one FO mark)
        marks = {(5, 14): "FI", (5, 19): "FO", (5, 27): "FO"}
        texts = {"2026/12/05": "FI", "2026/12/10": "FO", "2026/12/18": "FO"}
        texts.update({f"2026/12/{n:02d}": "task" for n in range(11, 18)})
        clears, sets, _ = self.plan([], remove=[("2026/12/05", "2026/12/10")], marks=marks, texts=texts)
        self.assertEqual([(5, 14)], clears)                      # the FO on 12/10 still ends one trip and starts the next

    def test_cells_with_tasks_or_red_cells_are_conflicts(self):
        clears, sets, report = self.plan([("2026/12/05", "2026/12/10")], texts={"2026/12/05": "task A"}, blocked={(5, 19)})
        self.assertEqual([], sets)
        self.assertEqual(2, sum(1 for r in report if r.startswith("CONFLICT")))

    def test_one_day_trips_and_unknown_workers_are_reported_not_written(self):
        _c, sets, report = self.plan([("2026/12/05", "2026/12/05")])
        self.assertEqual([], sets)
        self.assertTrue(any("one-day trip" in r for r in report))
        _c, sets, report = w.plan_trips({"nobody": [("2026/12/05", "2026/12/10")]}, {}, self.ROWS, self.COLS, {}, {}, lambda r, c: False, "2026/12/01")
        self.assertTrue(any("no row in the sheet" in r for r in report))



if __name__ == "__main__":
    unittest.main()
