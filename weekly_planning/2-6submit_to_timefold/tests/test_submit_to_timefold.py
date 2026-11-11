import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import submit_to_timefold as sub   # noqa: E402
import timefold_api as api         # noqa: E402
from fake_server import FakeApi    # noqa: E402

SCHEDULE = "schedule:\n  plan_range: {start_date: 2026/01/01, end_date: 2026/02/01}\n  workflow_task_list: []\n"
ENV = "environment:\n  worker_list: []\n"


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sub.main(args)
    return code, out.getvalue(), err.getvalue()


class Params(unittest.TestCase):
    def test_defaults_are_stage2_on_and_one_hour_in_total(self):
        p = api.build_params()
        self.assertEqual({"stage1_minutes": 45, "stage2_enabled": True, "stage2_minutes": 15, "allow_cut_fixed_trip": False}, p)
        self.assertEqual(60, api.time_budget_minutes(p))

    def test_options_win_and_no_stage2_removes_its_time(self):
        p = api.build_params(stage1_minutes=30, stage2_enabled=False, random_seed=7, allow_cut_fixed_trip=True)
        self.assertEqual((30, False, 7, True), (p["stage1_minutes"], p["stage2_enabled"], p["random_seed"], p["allow_cut_fixed_trip"]))
        self.assertEqual(30, api.time_budget_minutes(p))

    def test_params_file_is_layered_between_the_defaults_and_the_options(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "own.yaml"
            f.write_text("stage1_minutes: 5\nstage2_minutes: 3\nstage1_unimproved_seconds: 20\n", encoding="utf-8")
            p = api.build_params(str(f), stage2_minutes=9)
        self.assertEqual((5, 9, 20, True), (p["stage1_minutes"], p["stage2_minutes"], p["stage1_unimproved_seconds"], p["stage2_enabled"]))

    def test_bad_values_are_refused(self):
        for bad in ({"stage1_minutes": 0}, {"stage2_minutes": -3}):
            with self.assertRaises(api.ApiError):
                api.build_params(**bad)

    def test_params_yaml_round_trips(self):
        import yaml
        p = api.build_params(random_seed=3)
        self.assertEqual(p, yaml.safe_load(api.params_yaml(p)))


class Submit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.sched = self.dir / "03_Schedule.yaml"
        self.env = self.dir / "03_EnvConfig.yaml"
        self.sched.write_text(SCHEDULE, encoding="utf-8")
        self.env.write_text(ENV, encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def args(self, url, *extra):
        return ["--schedule", str(self.sched), "--env", str(self.env), "--mode", "local", "--api-url", url, *extra]

    def test_the_three_inputs_are_sent_under_the_names_the_solver_reads(self):
        with FakeApi() as srv:
            code, out, _ = run(self.args(srv.url, "--run-id", "test_run_1"))
        self.assertEqual(0, code)
        up = srv.uploads[0]
        self.assertEqual("test_run_1", up["runId"])
        self.assertEqual({"env", "sched", "params"}, set(up["files"]))
        self.assertEqual("EnvConfig.yaml", up["files"]["env"][0])
        self.assertEqual("Schedule.yaml", up["files"]["sched"][0])
        self.assertEqual("SolverParams.yaml", up["files"]["params"][0])
        self.assertEqual(self.sched.read_bytes(), up["files"]["sched"][1])       # the file is sent byte for byte
        self.assertIn(b"stage2_enabled: true", up["files"]["params"][1])
        self.assertIn("up to 60 minutes", out)

    def test_run_record_and_params_copy_are_written_next_to_the_schedule(self):
        with FakeApi() as srv:
            run(self.args(srv.url, "--run-id", "test_run_2", "--stage1-minutes", "20", "--seed", "5"))
        info = json.loads((self.dir / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(("test_run_2", "local", srv.url), (info["runId"], info["mode"], info["apiUrl"]))
        self.assertEqual(str(self.env.resolve()), info["envFile"])
        self.assertEqual((20, 5), (info["params"]["stage1_minutes"], info["params"]["random_seed"]))
        self.assertTrue((self.dir / "SolverParams.yaml").is_file())

    def test_out_dir_is_used_when_given(self):
        out_dir = self.dir / "elsewhere"
        with FakeApi() as srv:
            run(self.args(srv.url, "--out-dir", str(out_dir)))
        self.assertTrue((out_dir / "run_info.json").is_file())

    def test_dry_run_sends_nothing_and_writes_nothing(self):
        with FakeApi() as srv:
            code, out, _ = run(self.args(srv.url, "--dry-run"))
        self.assertEqual((0, []), (code, srv.uploads))
        self.assertIn("Nothing sent", out)
        self.assertFalse((self.dir / "run_info.json").exists())

    def test_real_mode_is_not_available_yet(self):
        code, _out, err = run(["--schedule", str(self.sched), "--env", str(self.env)])          # mode defaults to real
        self.assertEqual(2, code)
        self.assertIn("not available yet", err)

    def test_swapped_files_are_caught_before_anything_is_sent(self):
        with FakeApi() as srv:
            code, _out, err = run(["--schedule", str(self.env), "--env", str(self.sched), "--mode", "local", "--api-url", srv.url])
        self.assertEqual((1, []), (code, srv.uploads))
        self.assertIn("swapped", err)

    def test_server_not_running_gives_a_clear_message(self):
        code, _out, err = run(self.args("http://127.0.0.1:9"))
        self.assertEqual(1, code)
        self.assertIn("cannot reach", err)
        self.assertFalse((self.dir / "run_info.json").exists())

    def test_a_refused_run_is_reported_and_not_recorded(self):
        with FakeApi() as srv:
            srv.refuse_with = (400, {"error": "Missing required field: env"})
            code, _out, err = run(self.args(srv.url))
        self.assertEqual(1, code)
        self.assertIn("Missing required field", err)
        self.assertFalse((self.dir / "run_info.json").exists())


if __name__ == "__main__":
    unittest.main()
