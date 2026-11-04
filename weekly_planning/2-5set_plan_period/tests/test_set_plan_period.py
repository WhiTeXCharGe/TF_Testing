import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import set_plan_period as sp   # noqa: E402


def task(tid, name, phases):
    return {"id": tid, "name": name, "phase_task_list": [
        {"id": f"{tid}p{i}", "start_date": s, "end_date": e, "operation_task_list": [{"id": f"{tid}p{i}o1"}]}
        for i, (s, e) in enumerate(phases, 1)]}


def row(worker, op, days, flex="Flexible"):
    return {"worker": worker, "operation_task": op, "start_date": days[0], "end_date": days[-1],
            "work_date_list": [{"date": d, "hour": 10} for d in days], "plan_flexibility": flex}


def schedule():
    return {"schedule": {
        "plan_range": {"start_date": "2026/01/01", "end_date": "2027/03/31"},
        "workflow_task_list": [task("e1", "530N00001A", [("2026/10/05", "2026/11/10"), ("2026/11/11", "2026/12/07")]),
                               task("e2", "530N00002A", [("2026/09/01", "2026/10/30")])],
        "misc_task_list": [{"id": "misc_other_1", "name": "x", "region": "r_other"}],
        "assignment_list": [
            row("w1", "e1p1o1", ["2026/10/05", "2026/10/06", "2026/10/07", "2026/10/20"]),
            row("w2", "e2p1o1", ["2026/10/01", "2026/10/02"]),
            row("w3", "e2p1o1", ["2026/10/21", "2026/10/22"]),
            row("w1", "misc_other_1", ["2026/10/01"], flex="Fixed")],
        "trip_list": [{"worker": "w1", "trips": [
            {"start_date": "2026/10/04", "end_date": "2026/10/21", "region": "r1", "plan_flexibility": "Flexible"}]}]}}


class PlanPeriod(unittest.TestCase):
    def test_start_is_the_later_of_earliest_start_and_today_plus_k(self):
        _doc, _r, info = sp.plan_period(schedule(), "2026/10/10", k=7, l=30)       # earliest 09/01, today+K 10/17
        self.assertEqual("2026/10/17", info["start_date"])
        _doc, _r, info = sp.plan_period(schedule(), "2026/08/01", k=7, l=30)       # today+K 08/08 is earlier than 09/01
        self.assertEqual("2026/09/01", info["start_date"])

    def test_end_is_the_latest_due_date_plus_l(self):
        doc, _r, info = sp.plan_period(schedule(), "2026/10/10", k=7, l=30)
        self.assertEqual("2027/01/06", info["end_date"])                              # 12/07 + 30
        self.assertEqual({"start_date": "2026/10/17", "end_date": "2027/01/06"}, doc["schedule"]["plan_range"])

    def test_only_assignments_before_start_are_fixed_and_crossing_rows_are_split(self):
        doc, _r, info = sp.plan_period(schedule(), "2026/10/10", k=7, l=30)       # start 10/17
        rows = doc["schedule"]["assignment_list"]
        w1 = [r for r in rows if r["worker"] == "w1" and r["operation_task"] == "e1p1o1"]
        self.assertEqual(2, len(w1))
        fixed = next(r for r in w1 if r["plan_flexibility"] == "Fixed")
        flex = next(r for r in w1 if r["plan_flexibility"] == "Flexible")
        self.assertEqual(("2026/10/05", "2026/10/07"), (fixed["start_date"], fixed["end_date"]))
        self.assertEqual(["2026/10/20"], [i["date"] for i in flex["work_date_list"]])
        self.assertEqual(("2026/10/20", "2026/10/20"), (flex["start_date"], flex["end_date"]))
        w2 = next(r for r in rows if r["worker"] == "w2")                          # completely before: whole row Fixed
        self.assertEqual("Fixed", w2["plan_flexibility"])
        w3 = next(r for r in rows if r["worker"] == "w3")                          # after start: untouched
        self.assertEqual("Flexible", w3["plan_flexibility"])
        self.assertEqual(5, info["fixed_days"])
        self.assertEqual(1, info["split_rows"])

    def test_the_input_is_not_modified_and_other_parts_are_kept(self):
        original = schedule()
        before = copy.deepcopy(original)
        doc, _r, _i = sp.plan_period(original, "2026/10/10")
        self.assertEqual(before, original)
        self.assertEqual(before["schedule"]["workflow_task_list"], doc["schedule"]["workflow_task_list"])
        self.assertEqual(before["schedule"]["misc_task_list"], doc["schedule"]["misc_task_list"])
        self.assertEqual(before["schedule"]["trip_list"], doc["schedule"]["trip_list"])

    def test_end_before_start_is_repaired_with_a_warning(self):
        doc, report, info = sp.plan_period(schedule(), "2027/06/01", k=7, l=30)    # today is after every due date
        self.assertTrue(any(r.startswith("WARN") for r in report))
        self.assertLess(info["start_date"], info["end_date"])

    def test_a_schedule_without_modules_is_an_error(self):
        with self.assertRaises(ValueError):
            sp.plan_period({"schedule": {"workflow_task_list": [], "assignment_list": []}}, "2026/10/10")


if __name__ == "__main__":
    unittest.main()
