import contextlib
import copy
import io
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import set_plan_period as sp   # noqa: E402
from common.yaml_io import dump_yaml, load_yaml   # noqa: E402


def task(tid, phases):
    return {"id": tid, "name": tid, "phase_task_list": [
        {"id": f"{tid}p{i}", "start_date": s, "end_date": e, "operation_task_list": [{"id": f"{tid}p{i}o1"}]}
        for i, (s, e) in enumerate(phases, 1)]}


def row(worker, op, days, flex="Flexible"):
    return {"worker": worker, "operation_task": op, "start_date": days[0], "end_date": days[-1],
            "work_date_list": [{"date": d, "hour": 10} for d in days], "plan_flexibility": flex}


def schedule():
    return {"schedule": {
        "plan_range": {"start_date": "2026/01/01", "end_date": "2027/03/31"},
        "workflow_task_list": [task("e1", [("2026/10/05", "2026/11/10"), ("2026/11/11", "2026/12/07")]),
                               task("e2", [("2026/09/01", "2026/10/30")])],
        "misc_task_list": [{"id": "misc_other_1", "name": "x", "region": "r_other"}],
        "assignment_list": [
            row("w1", "e1p1o1", ["2026/10/05", "2026/10/06", "2026/10/20"]),    # crosses the start
            row("w2", "e2p1o1", ["2026/10/01", "2026/10/02"]),                  # completely before
            row("w3", "e2p1o1", ["2026/10/21", "2026/10/22"]),                  # completely after
            row("w1", "misc_other_1", ["2026/10/01"], flex="Fixed")],
        "trip_list": []}}


ENV_TEXT = "environment:\n  region_list:\n  - id: r1\n    name: 東京\n"


class Calculation(unittest.TestCase):
    def test_start_is_the_later_of_the_earliest_phase_start_and_today_plus_seven_days(self):
        phases = [(date(2026, 9, 1), date(2026, 10, 30)), (date(2026, 10, 5), date(2026, 12, 7))]
        self.assertEqual(date(2026, 10, 17), sp._calculate_plan_period(phases, date(2026, 10, 10))[0])
        self.assertEqual(date(2026, 9, 1), sp._calculate_plan_period(phases, date(2026, 8, 1))[0])

    def test_end_is_the_latest_phase_end_plus_thirty_days(self):
        phases = [(date(2026, 9, 1), date(2026, 10, 30)), (date(2026, 10, 5), date(2026, 12, 7))]
        self.assertEqual(date(2027, 1, 6), sp._calculate_plan_period(phases, date(2026, 10, 10))[1])     # 12/07 + 30

    def test_the_period_is_at_least_thirty_days_long(self):
        phases = [(date(2026, 9, 1), date(2026, 9, 10))]                      # due date + 30 is before start + 30
        start, end = sp._calculate_plan_period(phases, date(2026, 10, 10))      # start = 10/17
        self.assertEqual((date(2026, 10, 17), date(2026, 11, 16)), (start, end))

    def test_a_given_start_replaces_the_calculated_one(self):
        phases = [(date(2026, 9, 1), date(2026, 12, 7))]
        self.assertEqual((date(2026, 11, 1), date(2027, 1, 6)),
                         sp._calculate_plan_period(phases, date(2026, 10, 10), date(2026, 11, 1)))

    def test_without_phases_the_period_starts_in_seven_days_and_lasts_thirty(self):
        self.assertEqual((date(2026, 10, 17), date(2026, 11, 16)), sp._calculate_plan_period([], date(2026, 10, 10)))


class Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.sched_in = self.dir / "02_Schedule.yaml"
        self.env_in = self.dir / "02_EnvConfig.yaml"
        self.sched_out = self.dir / "out" / "03_Schedule.yaml"
        self.env_out = self.dir / "out" / "03_EnvConfig.yaml"
        dump_yaml(self.sched_in, schedule())
        self.env_in.write_bytes(ENV_TEXT.encode("utf-8"))

    def run_tool(self, plan_start=date(2026, 10, 17)):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            sp.set_plan_period(self.sched_in, self.env_in, self.sched_out, self.env_out, plan_start)
        return out.getvalue()

    def result(self):
        return load_yaml(self.sched_out)["schedule"]

    def test_the_plan_range_is_set_from_the_given_start_and_the_phase_end(self):
        self.run_tool()
        self.assertEqual({"start_date": "2026/10/17", "end_date": "2027/01/06"}, self.result()["plan_range"])

    def test_a_row_with_any_work_before_the_start_becomes_fixed_as_a_whole_rows_are_not_split(self):
        self.run_tool()
        rows = {(r["worker"], r["operation_task"]): r for r in self.result()["assignment_list"]}
        crossing = rows[("w1", "e1p1o1")]
        self.assertEqual("Fixed", crossing["plan_flexibility"])
        self.assertEqual(["2026/10/05", "2026/10/06", "2026/10/20"], [w["date"] for w in crossing["work_date_list"]])
        self.assertEqual("Fixed", rows[("w2", "e2p1o1")]["plan_flexibility"])
        self.assertEqual(4, len(self.result()["assignment_list"]))

    def test_a_row_that_starts_on_the_start_date_or_later_stays_flexible(self):
        self.run_tool()
        rows = {(r["worker"], r["operation_task"]): r for r in self.result()["assignment_list"]}
        self.assertEqual("Flexible", rows[("w3", "e2p1o1")]["plan_flexibility"])
        self.run_tool(plan_start=date(2026, 10, 21))                           # the first work day of w3 is the start
        rows = {(r["worker"], r["operation_task"]): r for r in self.result()["assignment_list"]}
        self.assertEqual("Flexible", rows[("w3", "e2p1o1")]["plan_flexibility"])

    def test_everything_else_is_kept(self):
        self.run_tool()
        src = schedule()["schedule"]
        self.assertEqual(src["workflow_task_list"], self.result()["workflow_task_list"])
        self.assertEqual(src["misc_task_list"], self.result()["misc_task_list"])
        self.assertEqual(src["trip_list"], self.result()["trip_list"])

    def test_the_input_files_are_not_changed_and_the_env_config_is_a_byte_copy(self):
        before = self.sched_in.read_bytes()
        self.run_tool()
        self.assertEqual(before, self.sched_in.read_bytes())
        self.assertEqual(ENV_TEXT.encode("utf-8"), self.env_in.read_bytes())
        self.assertEqual(ENV_TEXT.encode("utf-8"), self.env_out.read_bytes())

    def test_the_output_folder_is_created_and_the_paths_are_printed(self):
        text = self.run_tool()
        self.assertTrue(self.sched_out.exists())
        self.assertIn("03_Schedule.yaml", text)
        self.assertIn("03_EnvConfig.yaml", text)

    def test_a_schedule_without_the_wrapper_key_is_read_and_written_the_same_way(self):
        dump_yaml(self.sched_in, schedule()["schedule"])
        self.run_tool()
        doc = load_yaml(self.sched_out)
        self.assertNotIn("schedule", doc)
        self.assertEqual("2027/01/06", doc["plan_range"]["end_date"])

    def test_a_schedule_without_modules_still_gets_a_period(self):
        doc = schedule()
        doc["schedule"]["workflow_task_list"] = []
        doc["schedule"]["assignment_list"] = []
        dump_yaml(self.sched_in, doc)
        self.run_tool()
        self.assertEqual({"start_date": "2026/10/17", "end_date": "2026/11/16"}, self.result()["plan_range"])

    def test_the_same_path_for_input_and_output_is_refused(self):
        with self.assertRaises(ValueError):
            sp.set_plan_period(self.sched_in, self.env_in, self.sched_in, self.env_out)
        with self.assertRaises(ValueError):
            sp.set_plan_period(self.sched_in, self.env_in, self.sched_out, self.env_in)
        self.assertEqual(copy.deepcopy(schedule()), load_yaml(self.sched_in))   # nothing was written


class Validation(unittest.TestCase):
    def check(self, mutate, text):
        doc = schedule()
        mutate(doc["schedule"])
        with self.assertRaises(ValueError) as caught:
            sp._validated_schedule(doc)
        self.assertIn(text, str(caught.exception))

    def test_missing_keys_are_named(self):
        self.check(lambda s: s.pop("assignment_list"), "assignment_list")
        self.check(lambda s: s["plan_range"].pop("end_date"), "end_date")
        self.check(lambda s: s["workflow_task_list"][0].pop("phase_task_list"), "phase_task_list")

    def test_dates_are_checked_with_their_place(self):
        self.check(lambda s: s["workflow_task_list"][0]["phase_task_list"][0].update(start_date="2027/01/01"),
                   "start_date is after end_date")
        self.check(lambda s: s["workflow_task_list"][0]["phase_task_list"][0].update(end_date="x"), "invalid or missing date")
        self.check(lambda s: s["assignment_list"][0].update(end_date="2026/10/01"), "assignment_list[0]")

    def test_the_work_dates_must_be_inside_the_assignment(self):
        self.check(lambda s: s["assignment_list"][0]["work_date_list"].append({"date": "2026/12/31", "hour": 10}),
                   "outside assignment start_date/end_date")

    def test_an_empty_work_date_list_is_refused(self):
        self.check(lambda s: s["assignment_list"][0].update(work_date_list=[]), "must not be empty")

    def test_an_unknown_plan_flexibility_is_refused(self):
        self.check(lambda s: s["assignment_list"][0].update(plan_flexibility="Maybe"), "plan_flexibility")


class CommandLine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        dump_yaml(self.dir / "s.yaml", schedule())
        (self.dir / "e.yaml").write_bytes(ENV_TEXT.encode("utf-8"))

    def args(self, *extra):
        return ["--input-schedule", str(self.dir / "s.yaml"), "--input-env-config", str(self.dir / "e.yaml"),
                "--output-schedule", str(self.dir / "o_s.yaml"), "--output-env-config", str(self.dir / "o_e.yaml"), *extra]

    def test_main_writes_both_files_with_the_given_start(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = sp.main(self.args("--plan-start", "2026/11/01"))
        self.assertEqual(0, rc)
        self.assertEqual("2026/11/01", load_yaml(self.dir / "o_s.yaml")["schedule"]["plan_range"]["start_date"])
        self.assertTrue((self.dir / "o_e.yaml").exists())

    def test_an_invalid_start_date_is_an_argument_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sp.main(self.args("--plan-start", "next week"))

    def test_the_four_paths_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sp.main(["--input-schedule", str(self.dir / "s.yaml")])


if __name__ == "__main__":
    unittest.main()
