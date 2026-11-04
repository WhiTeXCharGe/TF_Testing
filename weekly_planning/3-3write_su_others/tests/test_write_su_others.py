import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import write_su_others as w   # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
