"""Tests of the decisions and the assembly of prepare_schedule.py (no Excel files needed).

Run from the tool folder:   python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_schedule as ps   # noqa: E402

CODE_A = "530N00001A"   # stays flexible
CODE_B = "530N00002A"   # planner moved a bar
CODE_C = "530N00003A"   # finished before today
CODE_D = "530N00004A"   # not a module of last week (new in the sheet)


def module(mid, code, fab="f1", ops=("p2o1",)):
    return {"id": mid, "name": code, "workflow": "wf_tool", "fab": fab, "description": None,
            "phase_task_list": [{"id": f"{mid}p2", "name": "HW", "phase": "p2", "start_date": "2026/10/01",
                                 "end_date": "2026/12/31",
                                 "operation_task_list": [{"id": f"{mid}{op}", "operation": op, "name": op,
                                                          "workload_hours": 100} for op in ops]}]}


def row(worker, op_task, days, flex="Flexible"):
    return {"worker": worker, "operation_task": op_task, "start_date": days[0], "end_date": days[-1],
            "work_date_list": [{"date": d, "hour": 10} for d in days], "plan_flexibility": flex}


def prev_env():
    return {"environment": {
        "workflow_list": [{"id": "wf_tool", "name": "Tool", "phase_list": [
            {"id": "p2", "name": "HW", "operation_list": [
                {"id": "p2o1", "name": "Mech", "work_hours": [10], "min_worker_num": 1, "max_worker_num": 8}]}]}],
        "fab_list": [{"id": "f1", "name": "F21", "region": "r1", "customer_company": "c1", "unavailable_dates": []}],
        "region_list": [{"id": "r1", "name": "USA", "max_stay_on": 80, "max_annual_stay": 240,
                         "stay_off_interval": 3, "unavailable_dates": []}],
        "customer_company_list": [{"id": "c1", "name": "TSMC", "unavailable_dates": []}],
        "worker_company_list": [{"id": "wc1", "name": "HumanEyes", "annual_overtime_limit": 10000,
                                 "monthly_overtime_limit": 10000, "unavailable_dates": []}],
        "transite_day_map": [], "affinity_tag": [{"id": "wct1", "weight": 2}],
        "worker_list": [
            {"id": "w007", "name": "山田 太郎", "worker_company": "wc1", "is_manager": True, "skill_map": {"p2o1": 3},
             "worker_type_by_operation": {"p2o1": "regular"}, "fab_suitability_map": [{"kind": "region", "suitability": {"r1": 2}}],
             "unavailable_dates": [], "affinity": ["wct1"]},
            {"id": "w008", "name": "佐藤 花子", "worker_company": "wc1", "is_manager": False, "skill_map": {"p2o1": 2},
             "worker_type_by_operation": {"p2o1": "spot"}, "fab_suitability_map": [], "unavailable_dates": [],
             "affinity": ["wct1"]}]}}


def prev_flexible():
    return {"schedule": {
        "plan_range": {"start_date": "2026/10/01", "end_date": "2026/12/31"},
        "workflow_task_list": [module("e1", CODE_A), module("e2", CODE_B), module("e3", CODE_C)],
        "assignment_list": [
            row("w007", "e1p2o1", ["2026/10/20", "2026/10/21", "2026/10/22"]),
            row("w008", "e2p2o1", ["2026/10/20", "2026/10/21"]),
            row("w007", "e3p2o1", ["2026/09/01", "2026/09/02"]),
            {"worker": "w007", "operation_task": "misc_other_1", "start_date": "2026/10/01", "end_date": "2026/10/01",
             "work_date_list": [{"date": "2026/10/01", "hour": 10}], "plan_flexibility": "Fixed"}],
        "trip_list": [{"worker": "w007", "trips": [
            {"start_date": "2026/10/19", "end_date": "2026/10/23", "region": "r1", "plan_flexibility": "Flexible", "description": None},
            {"start_date": "2026/11/10", "end_date": "2026/11/14", "region": "r1", "plan_flexibility": "Flexible", "description": None}]}]}}


def sheet(**over):
    k1, k2 = ps.name_key("山田 太郎"), ps.name_key("佐藤 花子")
    s = {
        "workers": [
            {"name": "山田 太郎", "company_name": "HumanEyes", "is_manager": True, "skill_map": {"p2o1": 3, "p2o2": 2},
             "worker_type_by_operation": {"p2o1": "regular", "p2o2": "regular"}, "unavailable": ["2026/10/30"], "description": None},
            {"name": "佐藤 花子", "company_name": "HumanEyes", "is_manager": False, "skill_map": {"p2o1": 2},
             "worker_type_by_operation": {"p2o1": "spot"}, "unavailable": [], "description": None},
            {"name": "鈴木 一郎", "company_name": "NewCo", "is_manager": False, "skill_map": {}, "worker_type_by_operation": {},
             "unavailable": [], "description": None}],
        "date_map": {
            (k1, "2026/10/20"): CODE_A, (k1, "2026/10/21"): CODE_A, (k1, "2026/10/22"): CODE_A,   # unchanged
            (k2, "2026/10/20"): CODE_B, (k2, "2026/10/22"): CODE_B,                                  # moved (21 -> 22)
            (k1, "2026/09/01"): CODE_C, (k1, "2026/09/02"): CODE_C,                                  # finished
            (k2, "2026/10/05"): "training",
            (k2, "2026/10/06"): "training",
            (k1, "2026/12/01"): CODE_D + " new tool"},
        "personal_map": {(k2, "2026/10/07"): "PB"},
        "color_map": {(k2, "2026/10/05"): "FFFF00"},
        "fifo": {k1: {"2026/10/19": "FI", "2026/10/23": "FO",       # same as the flexible trip -> stays flexible
                      "2026/11/09": "FI", "2026/11/15": "FO",       # FO moved -> Fixed trip
                      "2026/12/01": "FI", "2026/12/03": "FO"}},     # new -> Fixed trip
        "plan_range": {"start_date": "2026/01/01", "end_date": "2027/03/31"},
    }
    s.update(over)
    return s


SEIBAN = {CODE_A: {"customer": "TSMC", "fab": "F21", "region": "USA"},
          CODE_D: {"customer": "Micron", "fab": "F15", "region": "Taiwan"}}


class Decisions(unittest.TestCase):
    def test_unchanged_module_stays_flexible_changed_and_finished_become_fixed(self):
        prev = ps.read_prev_flexible(prev_flexible(), {"w007": "山田 太郎", "w008": "佐藤 花子"})
        cells = {}
        for (wk, d), t in sheet()["date_map"].items():
            c = ps.tool_code(t)
            if c:
                cells.setdefault(c, set()).add((wk, d))
        flexible, fixed = ps.decide_modules(prev["modules"], cells, "2026/10/10")
        self.assertEqual([CODE_A], flexible)
        self.assertIn("assignment changed", fixed[CODE_B])
        self.assertIn("1 worker-days", fixed[CODE_B])
        self.assertNotIn(CODE_A, fixed)
        # C is unchanged in the sheet but all its work ended before today
        self.assertIn("finished before today", fixed[CODE_C])

    def test_module_removed_from_the_sheet_is_fixed(self):
        prev = ps.read_prev_flexible(prev_flexible(), {"w007": "山田 太郎", "w008": "佐藤 花子"})
        flexible, fixed = ps.decide_modules(prev["modules"], {}, "2026/10/10")
        self.assertEqual([], flexible)
        self.assertEqual({CODE_A, CODE_B, CODE_C}, set(fixed))

    def test_a_module_without_assignments_and_without_bars_stays_flexible(self):
        flexible, fixed = ps.decide_modules({CODE_A: {"assign": set(), "last": None}}, {}, "2026/10/10")
        self.assertEqual(([CODE_A], {}), (flexible, fixed))

    def test_a_module_without_assignments_but_with_bars_is_fixed(self):
        flexible, fixed = ps.decide_modules({CODE_A: {"assign": set(), "last": None}},
                                            {CODE_A: {("k", "2026/10/20")}}, "2026/10/10")
        self.assertEqual([], flexible)
        self.assertIn(CODE_A, fixed)

    def test_sheet_cells_already_held_as_misc_task_of_the_module_are_not_a_change(self):
        # decoder7 turns outlier cells of a module into a misc task named after the module: same data must stay flexible
        cell = {(ps.name_key("山田 太郎"), "2026/10/20")}
        prev = {CODE_A: {"assign": set(), "last": None}}
        flexible, fixed = ps.decide_modules(prev, {CODE_A: cell}, "2026/10/10", {CODE_A: cell})
        self.assertEqual([CODE_A], flexible)
        flexible, fixed = ps.decide_modules(prev, {CODE_A: cell}, "2026/10/10", {CODE_A: set()})
        self.assertIn(CODE_A, fixed)

    def test_read_prev_flexible_collects_misc_cells_by_module_code(self):
        doc = prev_flexible()
        doc["schedule"]["misc_task_list"] = [{"id": "misc_other_9", "name": f"TSMCF21_SU3200_{CODE_D}_新規", "region": "r_other"}]
        doc["schedule"]["assignment_list"].append(
            {"worker": "w007", "operation_task": "misc_other_9", "start_date": "2026/10/02", "end_date": "2026/10/02",
             "work_date_list": [{"date": "2026/10/02", "hour": 10}], "plan_flexibility": "Fixed"})
        prev = ps.read_prev_flexible(doc, {"w007": "山田 太郎", "w008": "佐藤 花子"})
        self.assertEqual({CODE_D: {(ps.name_key("山田 太郎"), "2026/10/02")}}, prev["misc_cover"])

    def test_pair_fifo_and_orphans(self):
        pairs, warns = ps.pair_fifo({"2026/10/01": "FI", "2026/10/05": "FO", "2026/10/09": "FI", "2026/10/12": "FI",
                                     "2026/10/15": "FO", "2026/10/20": "FO", "2026/10/25": "FI"})
        self.assertEqual([("2026/10/01", "2026/10/05"), ("2026/10/12", "2026/10/15")], pairs)
        self.assertEqual(3, len(warns))
        self.assertTrue(any("10/09" in w and "next FI" in w for w in warns))
        self.assertTrue(any("FO 2026/10/20" in w for w in warns))
        self.assertTrue(any("FI 2026/10/25" in w for w in warns))

    def test_trips_kept_changed_and_fixed(self):
        prev = ps.read_prev_flexible(prev_flexible(), {"w007": "山田 太郎", "w008": "佐藤 花子"})
        k1 = ps.name_key("山田 太郎")
        dec = ps.decide_trips(prev["trips"], {k1: [("2026/10/19", "2026/10/23"), ("2026/11/09", "2026/11/15"),
                                                   ("2026/12/01", "2026/12/03")]})
        self.assertEqual([("2026/10/19", "2026/10/23")], [(t["start"], t["end"]) for t in dec["kept"]])
        self.assertEqual([("2026/11/10", "2026/11/14")], [(t["start"], t["end"]) for t in dec["changed"]])
        self.assertEqual([("2026/11/09", "2026/11/15"), ("2026/12/01", "2026/12/03")], [(s, e) for _, s, e, _ in dec["fixed"]])

    def test_prev_fixed_trip_whose_cells_are_still_there_stays_fixed_with_its_region(self):
        prev_trips = [{"worker_key": "k", "worker_id": "w1", "start": "2026/10/01", "end": "2026/10/05",
                       "region": "r9", "flex": "Fixed"}]
        dec = ps.decide_trips(prev_trips, {"k": [("2026/10/01", "2026/10/05")]})
        self.assertEqual([("k", "2026/10/01", "2026/10/05", "r9")], dec["fixed"])
        self.assertEqual([], dec["kept"] + dec["changed"])

    def test_trip_region_from_the_modules_between_fi_and_fo(self):
        days = {"k": {"2026/10/02": "M1", "2026/10/03": "M1", "2026/10/04": "M2", "2026/10/09": "M3"}}
        self.assertEqual(("r1", None), ps.infer_trip_region("k", "2026/10/01", "2026/10/05", {"k": {"2026/10/02": "M1"}}, {"M1": "r1"}))
        region, note = ps.infer_trip_region("k", "2026/10/01", "2026/10/05", days, {"M1": "r1", "M2": "r2"})
        self.assertEqual("r1", region)
        self.assertIn("several regions", note)
        self.assertEqual(None, ps.infer_trip_region("k", "2026/10/01", "2026/10/05", days, {})[0])

    def test_trip_region_falls_back_to_a_fab_name_in_the_label_and_never_guesses_between_regions(self):
        texts = {"k": {"2026/10/02": "TSMCF22_P3A_新規", "2026/10/09": "F15 other"}}
        fab_region = {"F22": "r2", "F15": "r3", "F2": "r9"}
        region, note = ps.infer_trip_region("k", "2026/10/01", "2026/10/05", {}, {}, texts, fab_region)
        self.assertEqual("r2", region)
        self.assertIn("fab name in the bar label", note)
        # both F22 (r2) and F15 (r3) inside -> no guess
        region, note = ps.infer_trip_region("k", "2026/10/01", "2026/10/12", {}, {}, texts, fab_region)
        self.assertIsNone(region)
        self.assertIn("several regions", note)
        # a module code of a known region wins over the label
        region, _ = ps.infer_trip_region("k", "2026/10/01", "2026/10/05", {"k": {"2026/10/03": "M1"}}, {"M1": "r7"}, texts, fab_region)
        self.assertEqual("r7", region)

    def test_fab_name_regions_skips_short_ambiguous_and_neutral_names(self):
        env = {"fab_list": [{"name": "F22", "region": "r2"}, {"name": "F22", "region": "r3"}, {"name": "F21", "region": "r1"},
                            {"name": "S3", "region": "r1"}, {"name": "Other", "region": "r_other"}, {"name": "Lehi", "region": "r1"}]}
        self.assertEqual({"F21": "r1", "LEHI": "r1"}, ps.fab_name_regions(env))


def days(first, last):
    from datetime import date, timedelta
    a, b = date(*map(int, first.split("/"))), date(*map(int, last.split("/")))
    return [(a + timedelta(n)).strftime("%Y/%m/%d") for n in range((b - a).days + 1)]


class FiFoPatterns(unittest.TestCase):
    WIN = ("2026/01/01", "2027/03/31")

    def test_read_mark_accepts_typos_case_and_notes_but_not_other_words(self):
        for text, mark in [("FI", "FI"), ("Fi", "FI"), ("ＦＩ", "FI"), (" fi ", "FI"), ("F1", "FI"), ("Fl", "FI"), ("FI(仮)", "FI"),
                           ("FO", "FO"), ("Fo", "FO"), ("F0", "FO"), ("FO（仮）", "FO")]:
            self.assertEqual(mark, ps.read_mark(text), text)
        for text in ["移動", "F22", "FC", "", None, 5, "FI 530N03359A"]:
            self.assertIsNone(ps.read_mark(text), text)

    def test_complete_pair_is_unchanged(self):
        trips, notes = ps.pair_fifo_repaired({"2026/03/01": "FI", "2026/03/20": "FO"}, [], self.WIN)
        self.assertEqual([("2026/03/01", "2026/03/20")], trips)
        self.assertEqual([], notes)

    def test_fi_without_fo_ends_on_the_last_busy_day_of_the_run(self):
        busy = days("2026/10/20", "2026/12/18") + days("2027/02/01", "2027/02/03")     # a long gap after 12/18
        trips, notes = ps.pair_fifo_repaired({"2026/10/19": "FI"}, busy, self.WIN)
        self.assertEqual([("2026/10/19", "2026/12/18")], trips)
        self.assertTrue(notes[0].startswith("INFERRED end 2026/12/18"))

    def test_fi_without_fo_and_nothing_busy_is_a_one_day_trip(self):
        trips, _ = ps.pair_fifo_repaired({"2026/10/19": "FI"}, [], self.WIN)
        self.assertEqual([("2026/10/19", "2026/10/19")], trips)

    def test_fi_followed_by_another_fi_is_closed_before_the_next_fi(self):
        busy = days("2026/10/21", "2026/11/18") + days("2026/11/26", "2026/12/23")
        marks = {"2026/10/20": "FI", "2026/11/25": "FI", "2026/12/30": "FI"}
        trips, _ = ps.pair_fifo_repaired(marks, busy, self.WIN, max_gap=14)
        self.assertEqual([("2026/10/20", "2026/11/18"), ("2026/11/25", "2026/12/23"), ("2026/12/30", "2026/12/30")], trips)

    def test_fo_without_fi_starts_on_the_first_busy_day_of_the_run(self):
        busy = days("2026/05/02", "2026/06/27")
        trips, notes = ps.pair_fifo_repaired({"2026/06/28": "FO"}, busy, self.WIN)
        self.assertEqual([("2026/05/02", "2026/06/28")], trips)
        self.assertTrue(notes[0].startswith("INFERRED start 2026/05/02"))

    def test_first_fo_of_the_sheet_means_the_trip_began_before_the_sheet(self):
        trips, _ = ps.pair_fifo_repaired({"2026/01/07": "FO"}, days("2026/01/03", "2026/01/06"), self.WIN)
        self.assertEqual([("2026/01/01", "2026/01/07")], trips)

    def test_fo_after_fo_starts_on_the_date_of_the_previous_fo(self):
        busy = days("2026/10/04", "2027/01/27")
        marks = {"2026/10/03": "FO", "2027/01/28": "FO"}
        trips, _ = ps.pair_fifo_repaired(marks, busy, self.WIN)
        # the first FO has nothing busy before it: a one-day trip; the second trip starts on the same date as the first FO
        self.assertEqual([("2026/10/03", "2026/10/03"), ("2026/10/03", "2027/01/28")], trips)

    def test_fi_taska_fo_taskb_fo_chain_shares_the_fo_date(self):
        """FI | TaskA ... | FO | TaskB ... | FO  ->  two trips, the second starts on the date of the first FO (not on the first TaskB day)."""
        busy = days("2026/10/21", "2026/10/29") + days("2026/10/31", "2026/11/09")
        marks = {"2026/10/20": "FI", "2026/10/30": "FO", "2026/11/10": "FO"}
        trips, notes = ps.pair_fifo_repaired(marks, busy, self.WIN)
        self.assertEqual([("2026/10/20", "2026/10/30"), ("2026/10/30", "2026/11/10")], trips)
        self.assertTrue(any("same date as the previous FO" in n for n in notes))

    def test_fo_chain_covers_a_weekend_between_the_fo_and_the_next_work(self):
        busy = days("2026/11/02", "2026/11/09")                      # FO on a Friday, work starts on the Monday (3 days later)
        trips, _ = ps.pair_fifo_repaired({"2026/10/30": "FO", "2026/11/10": "FO"}, busy, self.WIN)
        self.assertEqual(("2026/10/30", "2026/11/10"), trips[-1])

    def test_fo_chain_does_not_apply_when_the_work_starts_long_after_the_previous_fo(self):
        busy = days("2026/11/06", "2026/11/09")                      # 6 days after the first FO
        trips, _ = ps.pair_fifo_repaired({"2026/10/30": "FO", "2026/11/10": "FO"}, busy, self.WIN)
        self.assertEqual(("2026/11/06", "2026/11/10"), trips[-1])

    def test_fi_taska_fi_taskb_fo_chain_shares_the_fi_date(self):
        busy = days("2026/10/21", "2026/10/29") + days("2026/11/01", "2026/11/09")
        marks = {"2026/10/20": "FI", "2026/10/30": "FI", "2026/11/10": "FO"}
        trips, _ = ps.pair_fifo_repaired(marks, busy, self.WIN)
        self.assertEqual([("2026/10/20", "2026/10/30"), ("2026/10/30", "2026/11/10")], trips)

    def test_a_one_day_trip_between_two_fo_is_not_taken_for_a_duplicate_mark(self):
        trips, _ = ps.pair_fifo_repaired({"2026/10/30": "FO", "2026/11/01": "FO"}, ["2026/10/31"], self.WIN)
        self.assertEqual(2, len(trips))

    def test_double_marks_on_neighbouring_days_are_one_mark(self):
        trips, notes = ps.pair_fifo_repaired(
            {"2026/05/01": "FI", "2026/05/02": "FI", "2026/05/22": "FO", "2026/05/23": "FO"}, [], self.WIN)
        self.assertEqual([("2026/05/01", "2026/05/23")], trips)
        self.assertEqual(2, sum(1 for n in notes if n.startswith("duplicate")))

    def test_a_gap_larger_than_max_gap_stops_the_run(self):
        busy = days("2026/10/20", "2026/10/25") + days("2026/12/01", "2026/12/05")
        trips, _ = ps.pair_fifo_repaired({"2026/10/19": "FI"}, busy, self.WIN, max_gap=14)
        self.assertEqual([("2026/10/19", "2026/10/25")], trips)

    def test_prepare_keeps_every_trip_even_without_a_region_and_repairs_orphans(self):
        s = sheet()
        k1 = ps.name_key("山田 太郎")
        s["fifo"] = {k1: {"2026/12/10": "FI"}}                # no FO at all
        s["date_map"].update({(k1, d): "training" for d in days("2026/12/11", "2026/12/14")})
        out = ps.prepare(prev_flexible(), prev_env(), s, SEIBAN, "2026/10/10", default_trip_region="r_other")
        trips = {t["start_date"]: t for e in out["schedule"]["schedule"]["trip_list"] for t in e["trips"]}
        self.assertEqual("2026/12/14", trips["2026/12/10"]["end_date"])
        self.assertEqual("r_other", trips["2026/12/10"]["region"])
        self.assertEqual(0, out["stats"]["trips_skipped"])
        self.assertEqual(1, out["stats"]["trips_repaired"])


class CellNames(unittest.TestCase):
    """The text of a bar: a module whose bar was only renamed in SU_Others is changed too (becomes Fixed / misc)."""
    K1 = ps.name_key("山田 太郎")

    def prev_with_names(self, name="TSMC_" + CODE_A):
        doc = prev_flexible()
        doc["schedule"]["assignment_list"][0]["cell_name"] = name          # module A, w007, 10/20-10/22
        return doc

    def sheet_with_text(self, text):
        s = sheet()
        for d in ("2026/10/20", "2026/10/21", "2026/10/22"):
            s["date_map"][(self.K1, d)] = text
        return s

    def test_same_text_keeps_the_module_flexible(self):
        out = ps.prepare(self.prev_with_names(), prev_env(), self.sheet_with_text("TSMC_" + CODE_A), SEIBAN, "2026/10/10")
        self.assertIn(CODE_A, out["flexible_work"]["flexible_work"]["modules"])

    def test_only_the_text_changed_makes_the_module_fixed_and_the_misc_task_carries_the_new_text(self):
        out = ps.prepare(self.prev_with_names(), prev_env(), self.sheet_with_text("TSMC_" + CODE_A + "_renamed"), SEIBAN, "2026/10/10")
        self.assertNotIn(CODE_A, out["flexible_work"]["flexible_work"]["modules"])
        reason = next(line for line in out["report"] if f"FIXED module {CODE_A}" in line)
        self.assertIn("cell text changed", reason)
        self.assertIn("_renamed", reason)
        self.assertTrue(any(m["name"] == "TSMC_" + CODE_A + "_renamed" for m in out["schedule"]["schedule"]["misc_task_list"]))

    def test_text_only_differing_in_case_width_or_spaces_is_not_a_change(self):
        out = ps.prepare(self.prev_with_names("TSMC_" + CODE_A), prev_env(), self.sheet_with_text("ＴＳＭＣ _ " + CODE_A.lower()), SEIBAN, "2026/10/10")
        self.assertIn(CODE_A, out["flexible_work"]["flexible_work"]["modules"])

    def test_rows_without_a_cell_name_are_not_checked(self):
        out = ps.prepare(prev_flexible(), prev_env(), self.sheet_with_text("anything else " + CODE_A), SEIBAN, "2026/10/10")
        self.assertIn(CODE_A, out["flexible_work"]["flexible_work"]["modules"])

    def test_one_odd_cell_in_a_bar_is_not_a_change_but_the_most_common_text_is_what_counts(self):
        s = self.sheet_with_text("TSMC_" + CODE_A)
        s["date_map"][(self.K1, "2026/10/22")] = "TSMC_" + CODE_A + " (note)"
        out = ps.prepare(self.prev_with_names(), prev_env(), s, SEIBAN, "2026/10/10")
        self.assertIn(CODE_A, out["flexible_work"]["flexible_work"]["modules"])

    def test_flexible_rows_get_their_cell_name_from_the_sheet_and_it_is_also_kept_in_description(self):
        s = self.sheet_with_text("TSMC_" + CODE_A)
        out = ps.prepare(prev_flexible(), prev_env(), s, SEIBAN, "2026/10/10")      # previous rows have no cell_name yet
        row = next(r for r in out["schedule"]["schedule"]["assignment_list"] if r["operation_task"] == "e1p2o1")
        self.assertEqual("TSMC_" + CODE_A, row["cell_name"])
        self.assertEqual("TSMC_" + CODE_A, row["description"])

    def test_an_existing_description_is_not_overwritten(self):
        doc = prev_flexible()
        doc["schedule"]["assignment_list"][0]["description"] = "planner note"
        out = ps.prepare(doc, prev_env(), self.sheet_with_text("TSMC_" + CODE_A), SEIBAN, "2026/10/10")
        row = next(r for r in out["schedule"]["schedule"]["assignment_list"] if r["operation_task"] == "e1p2o1")
        self.assertEqual(("TSMC_" + CODE_A, "planner note"), (row["cell_name"], row["description"]))

    def test_read_prev_flexible_collects_the_names_per_worker_and_day(self):
        prev = ps.read_prev_flexible(self.prev_with_names("N1"), {"w007": "山田 太郎", "w008": "佐藤 花子"})
        self.assertEqual({(self.K1, "2026/10/20"): "N1", (self.K1, "2026/10/21"): "N1", (self.K1, "2026/10/22"): "N1"},
                         prev["modules"][CODE_A]["names"])

    def test_default_name_is_made_from_customer_fab_and_code(self):
        info = ps.module_name_info({"fab": "f1", "description": "d"}, prev_env()["environment"])
        self.assertEqual({"fab": "F21", "customer": "TSMC", "description": "d"}, info)
        self.assertEqual("TSMCF21_" + CODE_A + "_新規", ps.module_cell_name(CODE_A, info))
        self.assertEqual(CODE_A + "_新規", ps.module_cell_name(CODE_A, {}))


class FiFoDoesNotFixModules(unittest.TestCase):
    """A moved FI/FO changes the trip only: a module whose work in the sheet did not move stays flexible."""

    def test_moved_fi_fo_makes_a_fixed_trip_but_the_modules_inside_stay_flexible(self):
        s = sheet()
        k1 = ps.name_key("山田 太郎")
        # module A (unchanged work 10/20-10/22) lies inside last week's flexible trip 10/19-10/23; both FI and FO are moved now
        s["fifo"] = {k1: {"2026/10/17": "FI", "2026/10/26": "FO"}}
        out = ps.prepare(prev_flexible(), prev_env(), s, SEIBAN, "2026/10/10")
        fw = out["flexible_work"]["flexible_work"]
        self.assertIn(CODE_A, fw["modules"])                          # the module is still optimised
        self.assertEqual([], [t for t in fw["trips"] if t["start_date"] == "2026/10/19"])    # the moved trip is not flexible any more
        trips = [t for e in out["schedule"]["schedule"]["trip_list"] for t in e["trips"]]
        moved = next(t for t in trips if t["start_date"] == "2026/10/17")
        self.assertEqual(("2026/10/26", "Fixed"), (moved["end_date"], moved["plan_flexibility"]))
        self.assertTrue(any("CHANGED flexible trip" in line and "2026/10/19" in line for line in out["report"]))

    def test_unmoved_fi_fo_keeps_the_trip_flexible_and_the_flexible_work_lists_it(self):
        out = ps.prepare(prev_flexible(), prev_env(), sheet(), SEIBAN, "2026/10/10")
        fw = out["flexible_work"]["flexible_work"]
        self.assertEqual([{"worker": "w007", "start_date": "2026/10/19", "end_date": "2026/10/23", "region": "r1"}], fw["trips"])


class Assembly(unittest.TestCase):
    def setUp(self):
        self.out = ps.prepare(prev_flexible(), prev_env(), sheet(), SEIBAN, "2026/10/10")
        self.env = self.out["env"]["environment"]
        self.sch = self.out["schedule"]["schedule"]

    def test_flexible_work_lists_the_flexible_module_and_the_unchanged_trip(self):
        fw = self.out["flexible_work"]["flexible_work"]
        self.assertEqual([CODE_A], fw["modules"])
        self.assertEqual([{"worker": "w007", "start_date": "2026/10/19", "end_date": "2026/10/23", "region": "r1"}], fw["trips"])

    def test_worker_ids_of_known_workers_are_kept_and_new_workers_get_free_ids(self):
        ids = {w["name"]: w["id"] for w in self.env["worker_list"]}
        self.assertEqual("w007", ids["山田 太郎"])
        self.assertEqual("w008", ids["佐藤 花子"])
        self.assertEqual("w001", ids["鈴木 一郎"])

    def test_skills_for_unknown_operations_are_dropped_and_reported(self):
        w = next(w for w in self.env["worker_list"] if w["id"] == "w007")
        self.assertEqual({"p2o1": 3}, w["skill_map"])
        self.assertEqual({"p2o1": "regular"}, w["worker_type_by_operation"])
        self.assertTrue(any("operation p2o2" in line for line in self.out["report"]))
        # unavailable dates come from the sheet, fab suitability from last week's EnvConfig
        self.assertEqual([{"single": {"days": ["2026/10/30"]}}], w["unavailable_dates"])
        self.assertEqual([{"kind": "region", "suitability": {"r1": 2}}], w["fab_suitability_map"])

    def test_worker_missing_from_the_skill_file_keeps_last_weeks_skills(self):
        s = sheet()
        s["workers"][1].update(skill_map={}, worker_type_by_operation={}, has_skill_ref=False)   # 佐藤: not in the skill file
        s["workers"][2].update(has_skill_ref=False)                                                # 鈴木: new, no skills anywhere
        out = ps.prepare(prev_flexible(), prev_env(), s, SEIBAN, "2026/10/10")
        by_name = {w["name"]: w for w in out["env"]["environment"]["worker_list"]}
        self.assertEqual({"p2o1": 2}, by_name["佐藤 花子"]["skill_map"])
        self.assertEqual({"p2o1": "spot"}, by_name["佐藤 花子"]["worker_type_by_operation"])
        self.assertEqual({}, by_name["鈴木 一郎"]["skill_map"])
        self.assertTrue(any("not in the skill file" in line and "佐藤 花子" in line for line in out["report"]))

    def test_master_lists_keep_old_ids_and_add_new_ones(self):
        self.assertEqual("r1", next(r["id"] for r in self.env["region_list"] if r["name"] == "USA"))
        names = {r["name"]: r["id"] for r in self.env["region_list"]}
        self.assertEqual({"USA", "Other", "Taiwan"}, set(names))
        self.assertEqual("r_other", names["Other"])
        self.assertEqual("r2", names["Taiwan"])
        self.assertEqual({"f1", "f2"}, {f["id"] for f in self.env["fab_list"]})
        self.assertEqual("r2", next(f["region"] for f in self.env["fab_list"] if f["name"] == "F15"))
        self.assertEqual(["HumanEyes", "NewCo"], [c["name"] for c in self.env["worker_company_list"]])
        self.assertEqual("wc2", next(w["worker_company"] for w in self.env["worker_list"] if w["name"] == "鈴木 一郎"))
        pairs = {(t["from"], t["to"]) for t in self.env["transite_day_map"]}
        self.assertEqual(6, len(pairs))   # 3 regions, every ordered pair
        self.assertEqual(["wct1", "wct2"], [t["id"] for t in self.env["affinity_tag"]])

    def test_only_the_flexible_module_stays_and_its_rows_use_final_worker_ids(self):
        self.assertEqual([CODE_A], [t["name"] for t in self.sch["workflow_task_list"]])
        module_rows = [r for r in self.sch["assignment_list"] if r["operation_task"].startswith("e")]
        self.assertEqual(1, len(module_rows))
        self.assertEqual(("w007", "Flexible"), (module_rows[0]["worker"], module_rows[0]["plan_flexibility"]))

    def test_everything_else_is_misc_with_fixed_rows(self):
        misc = {m["name"]: m for m in self.sch["misc_task_list"]}
        self.assertEqual({CODE_B, CODE_C, "training", CODE_D + " new tool", "PB"}, set(misc))
        self.assertEqual("r_other", misc["training"]["region"])
        self.assertEqual("FFFF00", misc["training"]["color_code"])
        self.assertEqual("r2", misc[CODE_D + " new tool"]["region"])      # region of the module through 製番
        self.assertEqual("wf_personal_business", misc["PB"]["workflow"])
        misc_ids = {m["id"] for m in self.sch["misc_task_list"]}
        misc_rows = [r for r in self.sch["assignment_list"] if r["operation_task"] in misc_ids]
        self.assertTrue(misc_rows and all(r["plan_flexibility"] == "Fixed" for r in misc_rows))
        # a changed module keeps every bar of the sheet (worker-days), not the old plan
        b_id = misc[CODE_B]["id"]
        b_days = sorted(d["date"] for r in misc_rows if r["operation_task"] == b_id for d in r["work_date_list"])
        self.assertEqual(["2026/10/20", "2026/10/22"], b_days)
        # every row refers to a known task and a known worker
        task_ids = misc_ids | {o["id"] for t in self.sch["workflow_task_list"] for p in t["phase_task_list"]
                               for o in p["operation_task_list"]}
        worker_ids = {w["id"] for w in self.env["worker_list"]}
        for r in self.sch["assignment_list"]:
            self.assertIn(r["operation_task"], task_ids)
            self.assertIn(r["worker"], worker_ids)

    def test_trip_list_keeps_flexible_and_adds_fixed_trips_with_regions(self):
        trips = {t["worker"]: t["trips"] for t in self.sch["trip_list"]}
        self.assertEqual(["w007"], list(trips))
        got = [(t["start_date"], t["end_date"], t["region"], t["plan_flexibility"]) for t in trips["w007"]]
        # 11/09..11/15: no module between FI and FO -> no region -> skipped; 12/01..12/03: CODE_D is only on 12/01 (the FI day)
        self.assertEqual([("2026/10/19", "2026/10/23", "r1", "Flexible")], got)
        self.assertEqual(2, self.out["stats"]["trips_skipped"])
        self.assertTrue(any("Fixed trip" in line and "region could not be determined" in line for line in self.out["report"]))

    def test_default_trip_region_keeps_unresolvable_trips(self):
        out = ps.prepare(prev_flexible(), prev_env(), sheet(), SEIBAN, "2026/10/10", default_trip_region="r1")
        trips = out["schedule"]["schedule"]["trip_list"][0]["trips"]
        got = [(t["start_date"], t["end_date"], t["plan_flexibility"]) for t in trips]
        self.assertEqual([("2026/10/19", "2026/10/23", "Flexible"), ("2026/11/09", "2026/11/15", "Fixed"),
                          ("2026/12/01", "2026/12/03", "Fixed")], got)

    def test_fixed_trip_region_is_inferred_from_modules_between_fi_and_fo(self):
        s = sheet()
        k1 = ps.name_key("山田 太郎")
        s["date_map"][(k1, "2026/11/12")] = CODE_A + " site work"   # Fixed trip 11/09..11/15 now has a USA module inside
        out = ps.prepare(prev_flexible(), prev_env(), s, SEIBAN, "2026/10/10")
        trips = out["schedule"]["schedule"]["trip_list"][0]["trips"]
        self.assertIn(("2026/11/09", "2026/11/15", "r1", "Fixed"),
                      [(t["start_date"], t["end_date"], t["region"], t["plan_flexibility"]) for t in trips])

    def test_a_timefold_result_with_misc_tasks_is_a_valid_flexible_schedule(self):
        doc = prev_flexible()
        doc["schedule"]["misc_task_list"] = [{"id": "misc_other_1", "name": "x", "region": "r_other"}]
        out = ps.prepare(doc, prev_env(), sheet(), SEIBAN, "2026/10/10")
        self.assertEqual([CODE_A], out["flexible_work"]["flexible_work"]["modules"])

    def test_generated_yaml_round_trips(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "01_Schedule.yaml"
            ps.dump_yaml(p, self.out["schedule"])
            back = ps.load_yaml(p)
            self.assertEqual(self.out["schedule"], back)
            self.assertIn("trip_list", back["schedule"])


if __name__ == "__main__":
    unittest.main()
