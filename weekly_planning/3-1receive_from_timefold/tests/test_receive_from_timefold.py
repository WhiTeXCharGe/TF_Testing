import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(TOOLS / "2-6submit_to_timefold"))
sys.path.insert(0, str(TOOLS / "2-6submit_to_timefold" / "tests"))
import receive_from_timefold as rcv   # noqa: E402
from fake_server import FakeApi       # noqa: E402

RUNNING = {"runId": "r1", "status": "Running", "stage": 1, "progress": 0.35}
COMPLETED = {"runId": "r1", "status": "Completed", "stage": 2, "progress": 1}


def run(args, answer=None):
    out, err = io.StringIO(), io.StringIO()
    patch = mock.patch("builtins.input", return_value=answer) if answer is not None else contextlib.nullcontext()
    with patch, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rcv.main(args)
    return code, out.getvalue(), err.getvalue()


class Receive(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.env = self.dir / "03_EnvConfig.yaml"
        self.env.write_text("environment:\n  worker_list: []\n", encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def info(self, url, mode="local"):
        path = self.dir / "run_info.json"
        path.write_text(json.dumps({"runId": "r1", "mode": mode, "apiUrl": url, "envFile": str(self.env)}), encoding="utf-8")
        return str(path)

    def test_a_run_that_is_not_finished_says_so_with_stage_and_progress(self):
        with FakeApi() as srv:
            srv.statuses = [RUNNING]
            code, out, _ = run(["--run-info", self.info(srv.url)])
        self.assertEqual(1, code)
        self.assertIn("Still running: Running (stage 1, 35%)", out)
        self.assertFalse((self.dir / "04_Schedule.yaml").exists())

    def test_a_run_that_has_not_started_yet_says_it_is_waiting(self):
        with FakeApi() as srv:
            srv.statuses = [{"status": "Submitted", "progress": 0}]
            code, out, _ = run(["--run-info", self.info(srv.url)])
        self.assertEqual(1, code)
        self.assertIn("Waiting to start: Submitted", out)

    def test_finished_and_yes_downloads_the_schedule_and_keeps_the_env(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            code, out, _ = run(["--run-info", self.info(srv.url), "--yes"])
        self.assertEqual(0, code)
        self.assertIn("Done.", out)
        self.assertEqual(srv.result, (self.dir / "04_Schedule.yaml").read_bytes())
        self.assertEqual(self.env.read_bytes(), (self.dir / "04_EnvConfig.yaml").read_bytes())

    def test_finished_asks_and_y_downloads(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            code, _out, _ = run(["--run-info", self.info(srv.url)], answer="Y")
        self.assertEqual(0, code)
        self.assertTrue((self.dir / "04_Schedule.yaml").is_file())

    def test_finished_asks_and_n_ends_without_downloading(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            code, out, _ = run(["--run-info", self.info(srv.url)], answer="n")
        self.assertEqual(0, code)
        self.assertIn("Not downloaded", out)
        self.assertFalse((self.dir / "04_Schedule.yaml").exists())

    def test_no_terminal_to_answer_counts_as_no(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            with mock.patch("builtins.input", side_effect=EOFError):
                code, _out, _ = run(["--run-info", self.info(srv.url)])
        self.assertEqual(0, code)
        self.assertFalse((self.dir / "04_Schedule.yaml").exists())

    def test_no_flag_never_downloads(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            code, _out, _ = run(["--run-info", self.info(srv.url), "--no"])
        self.assertEqual(0, code)
        self.assertFalse((self.dir / "04_Schedule.yaml").exists())

    def test_out_dir_is_used_for_the_downloads(self):
        out_dir = self.dir / "week"
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            run(["--run-info", self.info(srv.url), "--yes", "--out-dir", str(out_dir)])
        self.assertTrue((out_dir / "04_Schedule.yaml").is_file())
        self.assertTrue((out_dir / "04_EnvConfig.yaml").is_file())

    def test_a_failed_run_shows_the_reason(self):
        with FakeApi() as srv:
            srv.statuses = [{"status": "Failed", "error": {"type": "InvalidInputData", "message": "trip_list invalid"}}]
            code, out, _ = run(["--run-info", self.info(srv.url), "--yes"])
        self.assertEqual(2, code)
        self.assertIn("InvalidInputData: trip_list invalid", out)
        self.assertFalse((self.dir / "04_Schedule.yaml").exists())

    def test_a_cancelled_run(self):
        with FakeApi() as srv:
            srv.statuses = [{"status": "Cancelled"}]
            code, out, _ = run(["--run-info", self.info(srv.url)])
        self.assertEqual(3, code)
        self.assertIn("Cancelled", out)

    def test_unknown_run_and_unreachable_server(self):
        with FakeApi() as srv:
            srv.status_code = 404
            srv.statuses = [{}]
            code, _out, err = run(["--run-info", self.info(srv.url)])
        self.assertEqual(4, code)
        self.assertIn("was not found", err)
        code, _out, err = run(["--run-info", self.info("http://127.0.0.1:9")])
        self.assertEqual(4, code)
        self.assertIn("cannot reach", err)

    def test_watch_checks_again_until_the_run_has_ended(self):
        with FakeApi() as srv:
            srv.statuses = [{"status": "Submitted", "progress": 0}, RUNNING, COMPLETED]
            with mock.patch("time.sleep") as sleep:
                code, out, _ = run(["--run-info", self.info(srv.url), "--watch", "5", "--yes"])
        self.assertEqual(0, code)
        self.assertEqual(2, sleep.call_count)
        self.assertIn("Running (stage 1, 35%) ... checking again in 5 s", out)
        self.assertTrue((self.dir / "04_Schedule.yaml").is_file())

    def test_real_mode_is_not_available_yet(self):
        code, _out, err = run(["--run-id", "r1"])
        self.assertEqual(4, code)
        self.assertIn("not available yet", err)

    def test_run_id_or_run_info_is_needed(self):
        with mock.patch("os.getcwd", return_value=str(self.dir)):
            code, _out, err = run([])
        self.assertEqual(5, code)
        self.assertIn("--run-info", err)

    def test_run_id_with_mode_local_works_without_a_run_record(self):
        with FakeApi() as srv:
            srv.statuses = [COMPLETED]
            code, _out, _ = run(["--run-id", "r1", "--mode", "local", "--api-url", srv.url, "--yes", "--out-dir", str(self.dir)])
        self.assertEqual(0, code)
        self.assertTrue((self.dir / "04_Schedule.yaml").is_file())
        self.assertFalse((self.dir / "04_EnvConfig.yaml").exists())


if __name__ == "__main__":
    unittest.main()
