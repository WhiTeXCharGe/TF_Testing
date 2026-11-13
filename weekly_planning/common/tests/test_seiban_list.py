"""Tests of seiban_list.py on small generated workbooks (no real files).

Run from the common folder:   python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.styles.colors import Color

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common import seiban_list as sl   # noqa: E402
from common.text import match_key      # noqa: E402

PLAN_HEADER = ["登録日時", "製番", "装置名称", "顧客", "Fab", "Region", "開始可能日", "希望納期", "工程選択",
               "工程1_開始可能日", "工程1_終了期限日", "工程2_開始可能日", "工程2_終了期限日"]


def solid(rgb):
    return PatternFill("solid", fgColor=rgb)


def plan_row(code="530N00001A", customer="Cust A", fab="Fab A", region="Region X", template="工程テンプレート1",
             p1=(datetime(2026, 12, 1), datetime(2026, 12, 10)), p2=(datetime(2026, 12, 5), datetime(2026, 12, 20))):
    return [datetime(2026, 11, 1), code, "Tool_X", customer, fab, region, None, None, template, p1[0], p1[1], p2[0], p2[1]]


TEMPLATE_HEADER = ["工程番号", "作業番号", "作業名", "工数", "最小人数", "最大人数"]
TEMPLATE_1 = [[1, 1, "Mech", 160, 2, 4], [1, 2, "Elec", 40, 1, 1], [2, 1, "Mech", 576, 2, 4], [2, 2, "Q/C", 64, 1, 2]]
TEMPLATE_2 = [[2, 1, "Mech", 576, 2, 4], [2, 2, "Elec", 144, 1, 1]]


class Book(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = Path(self.td.name) / "list.xlsx"

    def tearDown(self):
        self.td.cleanup()

    def make(self, plan_rows, fills=None, header=None, sites=None, templates=None, plan_sheet="計画対象", extra=None):
        wb = Workbook()
        wb.active.title = "製番リスト"                       # the big master sheet: must never be needed
        ws = wb.create_sheet(plan_sheet)
        ws.append(header or PLAN_HEADER)
        for r in plan_rows:
            ws.append(r)
        for row_no, fill in (fills or {}).items():
            ws.cell(row_no, (header or PLAN_HEADER).index("製番") + 1).fill = fill
        if sites is not None:
            s = wb.create_sheet("搬入先テーブル")
            s.append(["搬入先", "顧客", "Fab", "Region"])
            for r in sites:
                s.append(r)
        for number, rows in (templates if templates is not None else {1: TEMPLATE_1, 2: TEMPLATE_2}).items():
            t = wb.create_sheet(f"工程テンプレート{number}")
            t.append(TEMPLATE_HEADER)
            for r in rows:
                t.append(r)
        if extra:
            extra(wb)
        wb.save(self.path)
        return sl.read_seiban_list(self.path)


class RowColours(Book):
    def test_white_grey_pink_and_unknown(self):
        rows = [plan_row("530N00001A"), plan_row("530N00002A"), plan_row("530N00003A"), plan_row("530N00004A"), plan_row("530N00005A"),
                plan_row("530N00006A")]
        data = self.make(rows, fills={3: solid("FFD9D9D9"), 4: solid("FFFFDCE6"), 5: solid("FFFFFF00"), 6: solid("FFFFFFFF"),
                                       7: PatternFill("solid", fgColor=Color(theme=0))})
        self.assertEqual([sl.ADD, sl.KEEP, sl.DELETE, sl.UNKNOWN, sl.ADD, sl.ADD], [r.kind for r in data.rows])
        self.assertEqual(["none", "D9D9D9", "FFDCE6", "FFFF00", "FFFFFF", "theme:0"], [r.colour for r in data.rows])
        self.assertIn("not white, grey or pink", data.rows[3].problems[0])
        self.assertEqual([2, 3, 4, 5, 6, 7], [r.excel_row for r in data.rows])

    def test_rows_of_returns_one_kind(self):
        data = self.make([plan_row("530N00001A"), plan_row("530N00002A")], fills={3: solid("FFFFDCE6")})
        self.assertEqual(["530N00002A"], [r.code for r in data.rows_of(sl.DELETE)])
        self.assertEqual(["530N00001A"], [r.code for r in data.rows_of(sl.ADD)])

    def test_other_theme_colour_and_pattern_are_unknown_not_guessed(self):
        data = self.make([plan_row("530N00001A"), plan_row("530N00002A")],
                         fills={2: PatternFill("solid", fgColor=Color(theme=3)), 3: PatternFill("lightGray", fgColor="FFD9D9D9")})
        self.assertEqual([sl.UNKNOWN, sl.UNKNOWN], [r.kind for r in data.rows])
        self.assertEqual(["theme:3", "pattern:lightGray"], [r.colour for r in data.rows])


class RowFields(Book):
    def test_values_are_read_by_header_name_whatever_the_order(self):
        header = list(reversed(PLAN_HEADER))
        row = list(reversed(plan_row("530N00001A", customer=" Cust A ", fab="Fab　A ", region="Region X")))
        data = self.make([row], header=header)
        r = data.rows[0]
        self.assertEqual(("530N00001A", "Cust A", "Fab　A", "Region X", 1), (r.code, r.customer, r.fab, r.region, r.template))
        self.assertEqual(("2026/12/01", "2026/12/10"), r.phases[1])

    def test_seiban_is_normalised_and_bad_ones_are_reported(self):
        data = self.make([plan_row("５３０ｎ００００１ａ"), plan_row("TOOL-1"), plan_row(None)])
        self.assertEqual(["530N00001A", None, None], [r.code for r in data.rows])
        self.assertIn("not a module code", data.rows[1].problems[0])
        self.assertIn("seiban is empty", data.rows[2].problems[0])

    def test_empty_fab_stays_empty_and_numbers_become_text(self):
        data = self.make([plan_row(fab=None, region=7.0)])
        self.assertEqual(("", "7"), (data.rows[0].fab, data.rows[0].region))

    def test_template_choice(self):
        data = self.make([plan_row(template="工程テンプレート２"), plan_row(template=None), plan_row(template="other")])
        self.assertEqual([2, None, None], [r.template for r in data.rows])
        self.assertIn("no process template", data.rows[1].problems[0])
        self.assertIn("not readable", data.rows[2].problems[0])

    def test_dates_text_serial_blank_and_unreadable(self):
        row = plan_row(p1=("2026/12/01", 46000), p2=(None, "later"))
        data = self.make([row])
        r = data.rows[0]
        self.assertEqual(("2026/12/01", "2025/12/09"), r.phases[1])        # 46000 = a normal Excel date serial
        self.assertEqual((None, None), r.phases[2])
        self.assertEqual(["process 2 end: not a date: 'later'"], r.problems)

    def test_blank_rows_are_skipped_and_the_sheet_stops_after_a_long_blank_run(self):
        rows = [plan_row("530N00001A"), [None] * 13, plan_row("530N00002A")] + [[None] * 13] * 4 + [plan_row("530N00003A")]
        self.make(rows)
        data = sl.read_seiban_list(self.path, blank_run_limit=3)
        self.assertEqual(["530N00001A", "530N00002A"], [r.code for r in data.rows])
        data = sl.read_seiban_list(self.path)
        self.assertEqual(3, len(data.rows))

    def test_every_process_column_of_the_sheet_is_read(self):
        header = PLAN_HEADER + ["工程3_開始可能日", "工程3_終了期限日"]
        row = plan_row() + [datetime(2027, 1, 1), datetime(2027, 1, 9)]
        data = self.make([row], header=header)
        self.assertEqual([1, 2, 3], sorted(data.rows[0].phases))
        self.assertEqual(("2027/01/01", "2027/01/09"), data.rows[0].phases[3])


class WorkbookErrors(Book):
    def test_missing_plan_sheet(self):
        with self.assertRaisesRegex(sl.SeibanListError, "not found"):
            self.make([plan_row()], plan_sheet="other")

    def test_missing_column_is_named(self):
        header = [h for h in PLAN_HEADER if h != "Region"]
        with self.assertRaisesRegex(sl.SeibanListError, "Region"):
            self.make([], header=header)

    def test_no_process_date_columns(self):
        header = PLAN_HEADER[:9]
        with self.assertRaisesRegex(sl.SeibanListError, "process date columns"):
            self.make([], header=header)


class Sites(Book):
    SITES = [["Site_A", "Cust A", "Fab A", "Region X"], ["Site_B", "Cust B", None, "Region Y"], ["Site_C", "Cust C", "F６", "Region Z"]]

    def test_lookup_ignores_spaces_width_and_case(self):
        data = self.make([plan_row()], sites=self.SITES)
        self.assertEqual("Region X", data.site_region("CUST A ", "fab　a"))
        self.assertEqual("Region Z", data.site_region("Cust C", "F6"))
        self.assertEqual("Region Y", data.site_region("Cust B", ""))
        self.assertIsNone(data.site_region("Cust B", "Fab A"))
        self.assertEqual([], data.warnings)

    def test_duplicate_site_with_another_region_warns_and_the_first_wins(self):
        data = self.make([plan_row()], sites=self.SITES + [["Site_A2", "Cust A", "Fab A", "Region Q"]])
        self.assertEqual("Region X", data.site_region("Cust A", "Fab A"))
        self.assertTrue(any("appears twice" in w for w in data.warnings))

    def test_missing_site_sheet_is_only_a_warning(self):
        data = self.make([plan_row()])
        self.assertEqual([], data.sites)
        self.assertIsNone(data.site_region("Cust A", "Fab A"))
        self.assertTrue(any("搬入先テーブル" in w for w in data.warnings))


class Templates(Book):
    def test_operations_phases_and_work_name(self):
        data = self.make([plan_row()])
        t1, t2 = data.templates[1], data.templates[2]
        self.assertTrue(t1.valid and t2.valid)
        self.assertEqual([1, 2], t1.phases())
        self.assertEqual([2], t2.phases())                                  # template 2 has no process 1
        op = t1.ops_of(2)[1]
        self.assertEqual((2, 2, "QC", 64, 1, 2), (op.phase, op.op, op.name, op.hours, op.min_workers, op.max_workers))
        self.assertEqual([("Mech", 2, 4), ("Elec", 1, 1)], [(o.name, o.min_workers, o.max_workers) for o in t2.ops_of(2)])

    def test_template_numbers_come_from_the_sheet_names(self):
        data = self.make([plan_row()], templates={3: TEMPLATE_2, 12: TEMPLATE_1})
        self.assertEqual([3, 12], sorted(data.templates))
        self.assertEqual("工程テンプレート12", data.templates[12].sheet)

    def test_problems_are_kept_per_template_and_make_it_invalid(self):
        bad = [[1, 1, "Mech", "many", 1, 2], [1, 2, "Elec", 40, 3, 1], [2, 1, "Mech", 10, 1, 2], [2, 1, "Mech", 10, 1, 2],
               [None, 1, "Elec", 5, 1, 1], [3, 1, "", 5, 1, 1]]
        data = self.make([plan_row()], templates={1: bad, 2: TEMPLATE_2})
        t = data.templates[1]
        self.assertFalse(t.valid)
        text = " | ".join(t.problems)
        for part in ("row 2: 工数", "row 3: workers", "row 5: process 2 work 1 appears twice", "row 6: 工程番号", "row 7: 作業名"):
            self.assertIn(part, text)
        self.assertEqual(1, len(t.ops))                                     # the one good row is still read
        self.assertTrue(data.templates[2].valid)                             # another template is not affected

    def test_missing_column_and_empty_template(self):
        def extra(wb):
            ws = wb["工程テンプレート2"]
            ws.delete_cols(6)
        data = self.make([plan_row()], templates={1: [], 2: TEMPLATE_2}, extra=extra)
        self.assertFalse(data.templates[1].valid)
        self.assertIn("no operations", data.templates[1].problems[0])
        self.assertIn("最大人数", data.templates[2].problems[0])

    def test_no_template_sheet_is_a_warning(self):
        data = self.make([plan_row()], templates={})
        self.assertEqual({}, data.templates)
        self.assertTrue(any("工程テンプレート" in w for w in data.warnings))

    def test_work_name_normalising(self):
        for text, name in (("Q/C", "QC"), (" q/c ", "QC"), ("Ｑ／Ｃ", "QC"), ("Mech", "Mech"), ("M ech", "Mech")):
            self.assertEqual(name, sl.normalise_work_name(text), text)


class MatchKey(unittest.TestCase):
    def test_key_ignores_spaces_width_and_case(self):
        self.assertEqual(match_key("Cust"), match_key("Cust "))
        self.assertEqual(match_key("Cust A"), match_key("Cust\u3000A"))
        self.assertEqual(match_key("A6"), match_key("\uff21\uff16"))             # full-width A and 6
        self.assertEqual(match_key("Alpha"), match_key("ALPHA"))
        self.assertNotEqual(match_key("A21"), match_key("A22"))
        self.assertEqual("", match_key(None))


if __name__ == "__main__":
    unittest.main()
