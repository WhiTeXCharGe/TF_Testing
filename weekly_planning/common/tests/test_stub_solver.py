"""stub_solver: the solver stand-in speaks the contract that timefold_api uses."""
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common import timefold_api as api   # noqa: E402
from common.testing import stub_solver   # noqa: E402


class Stub(unittest.TestCase):
    def serve(self, seconds=0.0, result=None, fail=False):
        runs = stub_solver.Runs(self.root / "runs", seconds, result, fail)
        server = ThreadingHTTPServer(("127.0.0.1", 0), stub_solver.make_handler(runs))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        return api.Connection("local", f"http://127.0.0.1:{server.server_address[1]}")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.env, self.sched = self.root / "e.yaml", self.root / "s.yaml"
        self.env.write_text("environment: {}\n", encoding="utf-8")
        self.sched.write_text("schedule: {}\n" + "x: 1\n" * 200000, encoding="utf-8")     # several MB: a large answer must arrive whole

    def test_a_finished_run_gives_back_the_schedule_that_was_sent(self):
        conn = self.serve()
        answer = api.submit(conn, self.env, self.sched, "stage1_minutes: 1\n", "run_a")
        self.assertEqual({"runId": "run_a", "status": "Submitted"}, answer)
        self.assertEqual("Completed", api.get_status(conn, "run_a")["status"])
        self.assertEqual(self.sched.read_bytes(), api.download(conn, "run_a"))

    def test_a_run_goes_through_running_and_cannot_be_downloaded_before_the_end(self):
        conn = self.serve(seconds=60)
        api.submit(conn, self.env, self.sched, None, "run_b")
        state = api.get_status(conn, "run_b")
        self.assertIn(state["status"], ("Submitted", "Running"))
        with self.assertRaises(api.ApiError) as ctx:
            api.download(conn, "run_b")
        self.assertIn("409", str(ctx.exception))

    def test_a_fixed_result_file_and_a_failing_run(self):
        result = self.root / "result.yaml"
        result.write_bytes(b"solved: true\n")
        conn = self.serve(result=result)
        api.submit(conn, self.env, self.sched, None, "run_c")
        self.assertEqual(b"solved: true\n", api.download(conn, "run_c"))
        failing = self.serve(fail=True)
        api.submit(failing, self.env, self.sched, None, "run_d")
        self.assertEqual("Failed", api.get_status(failing, "run_d")["status"])

    def test_an_unknown_run_is_not_found_and_a_bad_run_id_gets_a_new_one(self):
        conn = self.serve()
        with self.assertRaises(api.ApiError) as ctx:
            api.get_status(conn, "nothing")
        self.assertIn("not found", str(ctx.exception))
        self.assertTrue(api.submit(conn, self.env, self.sched, None, "bad id!")["runId"].endswith("_stub"))
        api.health(conn)

    def test_make_run_creates_a_finished_run_for_a_hand_written_run_record(self):
        result = self.root / "result.yaml"
        result.write_bytes(b"solved: true\n")
        data = self.root / "kept"
        self.assertEqual(0, stub_solver.main(["--make-run", "hand_1", "--result", str(result), "--data-dir", str(data)]))
        runs = stub_solver.Runs(data, 20, result, False)
        server = ThreadingHTTPServer(("127.0.0.1", 0), stub_solver.make_handler(runs))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        conn = api.Connection("local", f"http://127.0.0.1:{server.server_address[1]}")
        self.assertEqual("Completed", api.get_status(conn, "hand_1")["status"])
        self.assertEqual(1, stub_solver.main(["--make-run", "bad id", "--result", str(result), "--data-dir", str(data)]))


if __name__ == "__main__":
    unittest.main()
