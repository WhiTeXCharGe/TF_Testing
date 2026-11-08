import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_flexible_schedule as ex   # noqa: E402


def row(worker, op, days, flex="Flexible"):
    return {"worker": worker, "operation_task": op, "start_date": days[0], "end_date": days[-1],
            "work_date_list": [{"date": d, "hour": 10} for d in days], "plan_flexibility": flex}


def trip(s, e, flex="Flexible", region="r1"):
    return {"start_date": s, "end_date": e, "region": region, "plan_flexibility": flex, "description": None}


def result():
    return {"schedule": {
        "plan_range": {"start_date": "2026/10/17", "end_date": "2027/01/06"},
        "workflow_task_list": [{"id": "e1", "name": "530N00001A", "fab": "f1", "phase_task_list": [
            {"id": "e1p1", "start_date": "2026/10/17", "end_date": "2026/11/10", "operation_task_list": [{"id": "e1p1o1"}]}]}],
        "misc_task_list": [{"id": "misc_other_1", "name": "x", "region": "r_other"}],
        "assignment_list": [row("w1", "e1p1o1", ["2026/10/20", "2026/10/21"]),
                            row("w2", "e1p1o1", ["2026/10/22"], flex="Fixed"),
                            row("w1", "misc_other_1", ["2026/10/25"], flex="Fixed")],
        "trip_list": [
            {"worker": "w1", "trips": [trip("2026/10/19", "2026/10/23"),            # module work inside: kept
                                       trip("2026/12/01", "2026/12/05"),            # Flexible but no module work: dropped
                                       trip("2026/10/24", "2026/10/28", flex="Fixed")]},   # Fixed (from the sheet): dropped
            {"worker": "w2", "trips": [trip("2026/10/22", "2026/10/22")]},          # one-day trip around w2's work: kept
            {"worker": "w9", "trips": [trip("2026/10/20", "2026/10/21")]}],         # no work of w9: dropped
        "solver_version": "x"}}


ENV = {"environment": {"worker_list": [{"id": "w1", "name": "A"}, {"id": "w2", "name": "B"}],
                       "fab_list": [{"id": "f1", "name": "F21", "customer_company": "c1"}],
                       "customer_company_list": [{"id": "c1", "name": "TSMC"}]}}


class Extract(unittest.TestCase):
    def setUp(self):
        self.doc, self.report, self.stats = ex.extract(result(), ENV)
        self.sch = self.doc["schedule"]

    def test_modules_and_module_assignments_are_kept_unchanged_and_misc_rows_are_left_out(self):
        src = result()["schedule"]
        self.assertEqual(src["workflow_task_list"], self.sch["workflow_task_list"])
        rows = [{k: v for k, v in r.items() if k != "cell_name"} for r in self.sch["assignment_list"]]
        self.assertEqual(src["assignment_list"][:2], rows)                           # both flexibilities, same order, only cell_name added
        self.assertEqual([], self.sch["misc_task_list"])
        self.assertEqual(1, self.stats["misc_or_other_rows_dropped"])

    def test_only_flexible_trips_with_module_work_inside_are_kept(self):
        trips = {e["worker"]: [(t["start_date"], t["end_date"]) for t in e["trips"]] for e in self.sch["trip_list"]}
        self.assertEqual({"w1": [("2026/10/19", "2026/10/23")], "w2": [("2026/10/22", "2026/10/22")]}, trips)
        self.assertEqual({"fixed": 1, "no module work inside": 2}, self.stats["trips_dropped"])

    def test_plan_range_is_kept_and_solver_version_is_not(self):
        self.assertEqual({"start_date": "2026/10/17", "end_date": "2027/01/06"}, self.sch["plan_range"])
        self.assertNotIn("solver_version", self.sch)

    def test_every_module_row_gets_a_cell_name_made_from_fab_and_customer_when_it_has_none(self):
        self.assertEqual(["TSMCF21_530N00001A_新規"] * 2, [r["cell_name"] for r in self.sch["assignment_list"]])
        self.assertEqual({"made from fab and customer": 2}, self.stats["cell_name"])

    def test_the_name_of_the_row_is_kept_cell_name_first_then_description(self):
        src = result()
        src["schedule"]["assignment_list"][0]["cell_name"] = "KEEP_ME"
        src["schedule"]["assignment_list"][1]["description"] = "TSMC_530N00001A_planner"        # carried by the solver
        doc, _r, stats = ex.extract(src, ENV)
        self.assertEqual(["KEEP_ME", "TSMC_530N00001A_planner"], [r["cell_name"] for r in doc["schedule"]["assignment_list"]])
        self.assertEqual({"from the result": 2}, stats["cell_name"])
        self.assertEqual("TSMC_530N00001A_planner", doc["schedule"]["assignment_list"][1]["description"])   # description untouched

    def test_input_is_not_modified(self):
        src = result()
        before = copy.deepcopy(src)
        ex.extract(src, ENV)
        self.assertEqual(before, src)

    def test_unknown_workers_and_modules_without_assignments_are_warned(self):
        src = result()
        src["schedule"]["workflow_task_list"].append({"id": "e2", "name": "530N00002A", "phase_task_list": [
            {"id": "e2p1", "operation_task_list": [{"id": "e2p1o1"}]}]})
        src["schedule"]["assignment_list"].append(row("w77", "e1p1o1", ["2026/10/30"]))
        _doc, report, _s = ex.extract(src, ENV)
        text = "\n".join(report)
        self.assertIn("w77", text)
        self.assertIn("module 530N00002A has no assignment", text)

    def test_the_flat_trip_form_is_read_too(self):
        src = result()
        src["schedule"]["trip_list"] = [{"worker": "w1", **trip("2026/10/19", "2026/10/23")}]
        doc, _r, _s = ex.extract(src, ENV)
        self.assertEqual([{"worker": "w1", "trips": [trip("2026/10/19", "2026/10/23")]}], doc["schedule"]["trip_list"])


if __name__ == "__main__":
    unittest.main()
