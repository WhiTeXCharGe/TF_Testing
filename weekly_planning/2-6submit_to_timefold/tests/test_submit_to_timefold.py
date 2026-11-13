import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))      # weekly_planning/ (the folder of common)
import submit_to_timefold as sub   # noqa: E402
from common import tf_config as tfc, timefold_api as api   # noqa: E402
from common.testing.fake_server import FakeApi   # noqa: E402

SCHEDULE = "schedule:\n  plan_range: {start_date: 2026/01/01, end_date: 2026/02/01}\n  workflow_task_list: []\n"
ENV = "environment:\n  worker_list: []\n"


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sub.main(args)
    return code, out.getvalue(), err.getvalue()


class DeliveredConfig(unittest.TestCase):
    def test_the_config_txt_of_the_tool_is_valid_and_lists_every_setting(self):
        cfg = tfc.load(str(HERE.parent / "config.txt"), None, tfc.SUBMIT_KEYS)
        self.assertEqual([], cfg.unknown)
        self.assertEqual(set(tfc.SUBMIT_KEYS), set(cfg.values))
        self.assertIn(cfg.text("mode"), api.MODES)


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
        # never read the real config.txt of the tool in a test
        patches = [mock.patch.object(sub, "CONFIG_FILE", self.dir / "no_such_config.txt"), mock.patch.object(api, "RETRY_DELAYS", (0, 0))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.td.cleanup()

    def config(self, text):
        path = self.dir / "config.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def args(self, url, *extra):
        return ["--schedule", str(self.sched), "--env", str(self.env), "--mode", "local", "--api-url", url, *extra]

    # ---- command line only (no config.txt)

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
        self.assertEqual(["/runSolver"], srv.paths)                                # local mode: no health check

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

    def test_no_mode_anywhere_is_a_clear_error(self):
        code, _out, err = run(["--schedule", str(self.sched), "--env", str(self.env)])
        self.assertEqual(1, code)
        self.assertIn("mode is not set", err)

    def test_swapped_files_are_caught_before_anything_is_sent(self):
        with FakeApi() as srv:
            code, _out, err = run(["--schedule", str(self.env), "--env", str(self.sched), "--mode", "local", "--api-url", srv.url])
        self.assertEqual((1, []), (code, srv.uploads))
        self.assertIn("swapped", err)

    def test_server_not_running_gives_a_clear_message(self):
        code, _out, err = run(self.args("http://127.0.0.1:9"))
        self.assertEqual(1, code)
        self.assertIn("cannot reach", err)
        self.assertIn("MockServer", err)
        self.assertFalse((self.dir / "run_info.json").exists())

    def test_a_refused_run_is_reported_and_not_recorded(self):
        with FakeApi() as srv:
            srv.refuse_with = (400, {"error": "Missing required field: env"})
            code, _out, err = run(self.args(srv.url))
        self.assertEqual(1, code)
        self.assertIn("Missing required field", err)
        self.assertFalse((self.dir / "run_info.json").exists())

    def test_a_run_id_the_server_would_replace_is_refused_up_front(self):
        with FakeApi() as srv:
            code, _out, err = run(self.args(srv.url, "--run-id", "week 45/final"))
        self.assertEqual((1, []), (code, srv.uploads))
        self.assertIn("run_id", err)

    # ---- config.txt

    def full_config(self, url, mode="local", extra=""):
        key = "local_url" if mode == "local" else "azure_url"
        return self.config(
            f"mode = {mode}\n{key} = {url}\nschedule = {self.sched}\nenv = {self.env}\nrun_id = cfg_run\n"
            f"stage1_minutes = 12\nstage2_enabled = false\nrandom_seed = 4   # fixed\n{extra}")

    def test_with_no_options_everything_comes_from_config_txt(self):
        with FakeApi() as srv:
            code, out, _ = run(["--config", self.full_config(srv.url)])
        self.assertEqual(0, code)
        up = srv.uploads[0]
        self.assertEqual("cfg_run", up["runId"])
        sent = up["files"]["params"][1].decode("utf-8")
        self.assertIn("stage1_minutes: 12", sent)
        self.assertIn("stage2_enabled: false", sent)
        self.assertIn("random_seed: 4", sent)
        self.assertIn("up to 12 minutes", out)
        info = json.loads((self.dir / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(("cfg_run", "local"), (info["runId"], info["mode"]))

    def test_command_line_replaces_config_txt(self):
        with FakeApi() as srv:
            run(["--config", self.full_config(srv.url), "--run-id", "cli_run", "--stage1-minutes", "7", "--seed", "9"])
        up = srv.uploads[0]
        self.assertEqual("cli_run", up["runId"])
        self.assertIn(b"stage1_minutes: 7", up["files"]["params"][1])
        self.assertIn(b"random_seed: 9", up["files"]["params"][1])

    def test_relative_paths_in_config_txt_are_relative_to_the_config_folder(self):
        with FakeApi() as srv:
            cfg = self.config(f"mode=local\nlocal_url={srv.url}\nschedule=03_Schedule.yaml\nenv=03_EnvConfig.yaml\nout_dir=sent\n")
            with mock.patch("os.getcwd", return_value=str(self.dir.parent)):
                code, _out, _ = run(["--config", cfg])
        self.assertEqual(0, code)
        self.assertTrue((self.dir / "sent" / "run_info.json").is_file())

    def test_a_missing_schedule_names_the_setting(self):
        code, _out, err = run(["--config", self.config("mode=local\n")])
        self.assertEqual(1, code)
        self.assertIn("schedule", err)
        self.assertIn("config", err)

    def test_a_bad_value_in_config_txt_is_reported(self):
        code, _out, err = run(["--config", self.full_config("http://x", extra="stage2_minutes = soon\n")])
        self.assertEqual(1, code)
        self.assertIn("stage2_minutes", err)

    def test_an_unknown_setting_is_mentioned_and_ignored(self):
        with FakeApi() as srv:
            code, _out, err = run(["--config", self.full_config(srv.url, extra="stage_1_minutes = 3\n"), "--dry-run"])
        self.assertEqual(0, code)
        self.assertIn("stage_1_minutes", err)

    # ---- online mode (Azure API controller; the fake server speaks the same contract)

    def test_online_mode_checks_the_server_first_and_sends_to_azure_url(self):
        with FakeApi() as srv:
            code, out, _ = run(["--config", self.full_config(srv.url + "/", mode="online")])
            self.assertEqual(["/health", "/runSolver"], srv.paths)
            self.assertEqual(1, len(srv.uploads))
        self.assertEqual(0, code)
        self.assertIn("(online mode)", out)
        info = json.loads((self.dir / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(("online", srv.url), (info["mode"], info["apiUrl"]))

    def test_online_mode_waits_for_a_server_that_is_still_starting(self):
        with FakeApi() as srv:
            srv.flaky = {"/health": [503, 502]}                      # an idle Container App answers like this while it wakes up
            code, _out, _ = run(["--config", self.full_config(srv.url, mode="online")])
            self.assertEqual((0, 1), (code, srv.health_calls))
            self.assertEqual(1, len(srv.uploads))

    def test_online_mode_gives_up_after_the_last_retry(self):
        with FakeApi() as srv:
            srv.flaky = {"/health": [503, 503, 503, 503]}
            code, _out, err = run(["--config", self.full_config(srv.url, mode="online")])
            self.assertEqual((1, []), (code, srv.uploads))
        self.assertIn("503", err)

    def test_local_mode_does_not_retry(self):
        with FakeApi() as srv:
            srv.flaky = {"/status/": [503]}
            with self.assertRaises(api.ApiError):
                api.get_status(api.Connection("local", srv.url), "r1")

    def test_online_mode_without_azure_url_or_with_the_placeholder_is_refused(self):
        code, _out, err = run(["--config", self.config(f"mode=online\nschedule={self.sched}\nenv={self.env}\n")])
        self.assertEqual(1, code)
        self.assertIn("azure_url is not set", err)
        code, _out, err = run(["--config", self.config(f"mode=online\nazure_url=https://<your-solver-api>.azurecontainerapps.io\nschedule={self.sched}\nenv={self.env}\n")])
        self.assertEqual(1, code)
        self.assertIn("placeholder", err)

    def test_online_mode_unreachable_points_at_the_settings_and_the_network(self):
        code, _out, err = run(["--config", self.full_config("http://127.0.0.1:9", mode="online")])
        self.assertEqual(1, code)
        self.assertIn("azure_url", err)
        self.assertIn("network", err)
        self.assertFalse((self.dir / "run_info.json").exists())

    def test_online_mode_does_not_send_when_the_address_is_not_the_api(self):
        with FakeApi() as srv:
            srv.flaky = {"/health": [404] * 4}
            code, _out, err = run(["--config", self.full_config(srv.url, mode="online")])
            self.assertEqual((1, []), (code, srv.uploads))
        self.assertIn("/health", err)

if __name__ == "__main__":
    unittest.main()
