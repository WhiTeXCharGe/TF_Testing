import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_flexible_schedule as ex   # noqa: E402
from common.yaml_io import dump_yaml, load_yaml   # noqa: E402


def row(worker, op, days, flex="Flexible"):
    return {"worker": worker, "operation_task": op, "start_date": days[0], "end_date": days[-1],
            "work_date_list": [{"date": d, "hour": 10} for d in days], "plan_flexibility": flex}


def trip(s, e, flex="Flexible", region="r1"):
    return {"start_date": s, "end_date": e, "region": region, "plan_flexibility": flex, "description": None}


def module(tid, name, end, color=None, fab="f1"):
    return {"id": tid, "name": name, "fab": fab, "phase_task_list": [
        {"id": f"{tid}p1", "start_date": "2026/10/17", "end_date": end,
         "operation_task_list": [{"id": f"{tid}p1o1", "color_code": color}]}]}


def result():
    return {"schedule": {
        "plan_range": {"start_date": "2026/10/17", "end_date": "2027/01/06"},
        "workflow_task_list": [module("e1", "530N00001A", "2026/11/10", color="AAAAAA")],
        "misc_task_list": [{"id": "misc_other_1", "name": "x", "region": "r_other"}],
        "assignment_list": [row("w1", "e1p1o1", ["2026/10/20", "2026/10/21"]),
                            row("w2", "e1p1o1", ["2026/10/22"], flex="Fixed"),
                            row("w1", "misc_other_1", ["2026/10/25"], flex="Fixed")],
        "trip_list": [
            {"worker": "w1", "trips": [trip("2026/10/19", "2026/10/23"),            # Flexible work of w1 inside: kept
                                       trip("2026/12/01", "2026/12/05"),            # no work inside: dropped
                                       trip("2026/10/24", "2026/10/28", flex="Fixed")]},   # Fixed trip: dropped
            {"worker": "w2", "trips": [trip("2026/10/22", "2026/10/22")]},          # only a Fixed row of w2 inside: dropped
            {"worker": "w9", "trips": [trip("2026/10/20", "2026/10/21")]}],         # no work of w9: dropped
        "solver_version": "x"}}


ENV = {"environment": {"worker_list": [{"id": "w1", "name": "A"}, {"id": "w2", "name": "B"}],
                       "fab_list": [{"id": "f1", "name": "FabA", "customer_company": "c1"}],
                       "customer_company_list": [{"id": "c1", "name": "CustA"}]}}


class Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.sched_in = self.dir / "04_Schedule.yaml"
        self.env_in = self.dir / "04_EnvConfig.yaml"
        self.out = self.dir / "FlexibleSchedule.yaml"
        dump_yaml(self.sched_in, result())
        dump_yaml(self.env_in, ENV)

    def run_tool(self):
        ex.create_flexible_schedule(self.sched_in, self.env_in, self.out)
        return load_yaml(self.out)["schedule"]


class Extract(Files):
    def test_only_the_expected_parts_are_written(self):
        sch = self.run_tool()
        self.assertEqual(["plan_range", "workflow_task_list", "assignment_list", "trip_list"], list(sch))
        self.assertEqual({"start_date": "2026/10/17", "end_date": "2027/01/06"}, sch["plan_range"])

    def test_module_rows_are_kept_in_order_with_both_flexibilities_and_misc_rows_are_left_out(self):
        sch = self.run_tool()
        src = result()["schedule"]["assignment_list"]
        rows = [{k: v for k, v in r.items() if k != "cell_name"} for r in sch["assignment_list"]]
        self.assertEqual(src[:2], rows)

    def test_the_workflow_tasks_are_kept(self):
        sch = self.run_tool()
        self.assertEqual("e1", sch["workflow_task_list"][0]["id"])
        self.assertEqual("530N00001A", sch["workflow_task_list"][0]["name"])

    def test_only_flexible_trips_with_flexible_module_work_of_the_same_worker_inside_are_kept(self):
        sch = self.run_tool()
        trips = {e["worker"]: [(t["start_date"], t["end_date"]) for t in e["trips"]] for e in sch["trip_list"]}
        self.assertEqual({"w1": [("2026/10/19", "2026/10/23")]}, trips)

    def test_a_work_day_on_the_first_or_last_day_of_the_trip_counts_as_inside(self):
        src = result()
        src["schedule"]["trip_list"] = [{"worker": "w1", "trips": [trip("2026/10/21", "2026/10/21"),     # last work day only
                                                                  trip("2026/10/10", "2026/10/20"),     # first work day at the end
                                                                  trip("2026/10/22", "2026/10/30")]}]   # one day after the work
        dump_yaml(self.sched_in, src)
        got = [(t["start_date"], t["end_date"]) for t in self.run_tool()["trip_list"][0]["trips"]]
        self.assertEqual([("2026/10/21", "2026/10/21"), ("2026/10/10", "2026/10/20")], got)

    def test_the_flat_trip_form_is_read_and_written_in_the_same_form(self):
        src = result()
        src["schedule"]["trip_list"] = [{"worker": "w1", **trip("2026/10/19", "2026/10/23")},
                                        {"worker": "w1", **trip("2026/12/01", "2026/12/05")}]
        dump_yaml(self.sched_in, src)
        self.assertEqual([{"worker": "w1", **trip("2026/10/19", "2026/10/23")}], self.run_tool()["trip_list"])

    def test_the_input_files_are_not_changed(self):
        before = (self.sched_in.read_bytes(), self.env_in.read_bytes())
        self.run_tool()
        self.assertEqual(before, (self.sched_in.read_bytes(), self.env_in.read_bytes()))

    def test_a_schedule_without_the_wrapper_key_is_written_without_it(self):
        dump_yaml(self.sched_in, result()["schedule"])
        ex.create_flexible_schedule(self.sched_in, self.env_in, self.out)
        doc = load_yaml(self.out)
        self.assertNotIn("schedule", doc)
        self.assertEqual(1, len(doc["workflow_task_list"]))


class CellNames(Files):
    def test_a_row_without_a_name_gets_one_made_from_customer_fab_and_module(self):
        sch = self.run_tool()
        self.assertEqual(["CustAFabA_530N00001A_新規"] * 2, [r["cell_name"] for r in sch["assignment_list"]])

    def test_the_cell_name_of_the_row_comes_first_then_the_description_that_the_solver_carries(self):
        src = result()
        src["schedule"]["assignment_list"][0]["cell_name"] = "KEEP_ME"
        src["schedule"]["assignment_list"][1]["description"] = "CustA_530N00001A_planner"
        dump_yaml(self.sched_in, src)
        rows = self.run_tool()["assignment_list"]
        self.assertEqual(["KEEP_ME", "CustA_530N00001A_planner"], [r["cell_name"] for r in rows])
        self.assertEqual("CustA_530N00001A_planner", rows[1]["description"])          # the description itself is not changed

    def test_a_blank_cell_name_counts_as_not_set(self):
        src = result()
        src["schedule"]["assignment_list"][0]["cell_name"] = "   "
        src["schedule"]["assignment_list"][0]["description"] = "FROM_DESCRIPTION"
        dump_yaml(self.sched_in, src)
        self.assertEqual("FROM_DESCRIPTION", self.run_tool()["assignment_list"][0]["cell_name"])


class Colors(Files):
    def colors(self, sch):
        return {t["id"]: {o["color_code"] for p in t["phase_task_list"] for o in p["operation_task_list"]}
                for t in sch["workflow_task_list"]}

    def test_a_module_without_a_color_gets_the_color_that_was_used_longest_ago(self):
        src = result()
        tasks = src["schedule"]["workflow_task_list"]
        tasks[0]["phase_task_list"][0]["end_date"] = "2026/10/10"                  # AAAAAA: last used 10/10
        tasks.append(module("e2", "530N00002A", "2026/11/10", color="BBBBBB"))       # BBBBBB: last used 11/10
        tasks.append(module("e3", "530N00003A", "2026/12/01"))                       # no color
        dump_yaml(self.sched_in, src)
        self.assertEqual({"e1": {"AAAAAA"}, "e2": {"BBBBBB"}, "e3": {"AAAAAA"}}, self.colors(self.run_tool()))

    def test_colors_of_module_misc_tasks_are_candidates_too_but_other_misc_tasks_are_not(self):
        src = result()
        src["schedule"]["workflow_task_list"].append(module("e3", "530N00003A", "2026/12/01"))
        src["schedule"]["misc_task_list"] += [
            {"id": "misc_a", "name": "530N00009A", "color_code": "CCCCCC"},         # a module: last work 09/01, older than AAAAAA
            {"id": "misc_b", "name": "personal", "color_code": "DDDDDD"}]           # not a module: never used
        src["schedule"]["assignment_list"] += [row("w1", "misc_a", ["2026/09/01"], flex="Fixed"),
                                              row("w1", "misc_b", ["2026/08/01"], flex="Fixed")]
        dump_yaml(self.sched_in, src)
        self.assertEqual({"e1": {"AAAAAA"}, "e3": {"CCCCCC"}}, self.colors(self.run_tool()))

    def test_two_new_modules_get_different_colors_while_colors_are_left(self):
        src = result()
        tasks = src["schedule"]["workflow_task_list"]
        tasks.append(module("e2", "530N00002A", "2026/11/20", color="BBBBBB"))
        tasks.append(module("e3", "530N00003A", "2026/12/01"))
        tasks.append(module("e4", "530N00004A", "2026/12/02"))
        dump_yaml(self.sched_in, src)
        got = self.colors(self.run_tool())
        self.assertEqual({"AAAAAA", "BBBBBB"}, got["e3"] | got["e4"])
        self.assertEqual(1, len(got["e3"]))
        self.assertNotEqual(got["e3"], got["e4"])

    def test_nothing_is_changed_when_no_module_has_a_color(self):
        src = result()
        src["schedule"]["workflow_task_list"][0]["phase_task_list"][0]["operation_task_list"][0]["color_code"] = None
        dump_yaml(self.sched_in, src)
        self.assertEqual({"e1": {None}}, self.colors(self.run_tool()))

    def test_existing_colors_are_kept(self):
        self.assertEqual({"e1": {"AAAAAA"}}, self.colors(self.run_tool()))


class Errors(Files):
    def fails_with(self, text, mutate):
        src = result()
        mutate(src["schedule"])
        dump_yaml(self.sched_in, src)
        with self.assertRaises(ValueError) as caught:
            ex.create_flexible_schedule(self.sched_in, self.env_in, self.out)
        self.assertIn(text, str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_a_missing_trip_list_is_an_error(self):
        self.fails_with("trip_list", lambda s: s.pop("trip_list"))

    def test_an_assignment_of_an_unknown_operation_is_an_error(self):
        self.fails_with("unknown OperationTask", lambda s: s["assignment_list"].append(row("w1", "nope", ["2026/10/30"])))

    def test_a_duplicate_operation_id_is_an_error(self):
        self.fails_with("duplicate", lambda s: s["workflow_task_list"].append(module("e1", "530N00001A", "2026/11/10")))

    def test_a_trip_with_a_bad_date_is_an_error(self):
        self.fails_with("invalid Trip date", lambda s: s["trip_list"][0]["trips"][0].update(start_date="x"))
        self.fails_with("start_date is after end_date", lambda s: s["trip_list"][0]["trips"][0].update(start_date="2027/12/01"))

    def test_a_trip_without_a_worker_is_an_error(self):
        self.fails_with("must have a worker", lambda s: s["trip_list"].append({"trips": []}))


class CommandLine(Files):
    def args(self):
        return ["--schedule", str(self.sched_in), "--env", str(self.env_in), "--output", str(self.out)]

    def test_main_writes_the_file_and_returns_zero(self):
        self.assertEqual(0, ex.main(self.args()))
        self.assertEqual(1, len(load_yaml(self.out)["schedule"]["workflow_task_list"]))

    def test_main_reports_an_error_on_stderr_and_returns_one(self):
        src = result()
        src["schedule"].pop("trip_list")
        dump_yaml(self.sched_in, src)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = ex.main(self.args())
        self.assertEqual(1, rc)
        self.assertIn("trip_list", err.getvalue())

    def test_a_missing_input_file_returns_one(self):
        self.sched_in.unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(1, ex.main(self.args()))


if __name__ == "__main__":
    unittest.main()
