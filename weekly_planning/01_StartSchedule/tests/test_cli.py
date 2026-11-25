"""The command line of start_schedule.py (with fake tools)."""
import contextlib
import io
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # 01_StartSchedule/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import start_schedule   # noqa: E402
from fake_tools import calls, make_tools   # noqa: E402


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class CommandLine(TempCase):
    def setUp(self):
        super().setUp()
        self.tools = make_tools(self.root / "tools")
        self.su, self.sb = self.root / "su.xlsm", self.root / "list.xlsm"
        self.su.write_bytes(b"1")
        self.sb.write_bytes(b"2")
        (self.root / "plan_assist.config.txt").write_text(
            f"tools_dir = {self.tools}\nwork_dir = {self.root / 'Work'}\nmaster_dir = {self.root / 'Master'}\nmode = local\n", encoding="utf-8")

    def args(self, *extra):
        return ["--config", str(self.root / "plan_assist.config.txt"), "--su-others", str(self.su), "--seiban-list", str(self.sb),
                "--date", "20261117", *extra]

    def run_main(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = start_schedule.main(argv)
        return rc, out.getvalue()

    def test_both_files_run_without_a_window_and_print_the_log(self):
        rc, text = self.run_main(self.args())
        self.assertEqual(0, rc)
        self.assertIn("週 20261117", text)
        self.assertIn("== [4/4] 2-6", text)
        self.assertEqual(["2-2-3", "2-4", "2-5", "2-6"], [c["step"] for c in calls(self.tools)])

    def test_no_submit_and_the_exit_code_of_a_failure(self):
        rc, text = self.run_main(self.args("--no-submit"))
        self.assertEqual(0, rc)
        self.assertNotIn("2-6", [c["step"] for c in calls(self.tools)])
        (self.tools / "fail_2-4").write_text("Error: boom", encoding="utf-8")
        rc, text = self.run_main(self.args("--no-submit", "--yes"))
        self.assertEqual(1, rc)
        self.assertIn("中断（2-4 製番リストの反映）", text)

    def test_a_sent_week_needs_yes(self):
        self.assertEqual(0, self.run_main(self.args())[0])
        rc, text = self.run_main(self.args())
        self.assertEqual(1, rc)
        self.assertIn("送信済み", text)
        self.assertEqual(0, self.run_main(self.args("--yes"))[0])

    def test_no_ui_without_files_is_an_error_and_a_bad_date_is_refused(self):
        rc, text = self.run_main(["--config", str(self.root / "plan_assist.config.txt"), "--no-ui"])
        self.assertEqual(1, rc)
        self.assertIn("選択されていません", text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            start_schedule.main(self.args()[:-2] + ["--date", "soon"])

    def test_a_missing_settings_file_is_reported(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = start_schedule.main(["--config", str(self.root / "none.txt"), "--no-ui"])
        self.assertEqual(1, rc)
        self.assertIn("設定ファイルが見つかりません", err.getvalue())

    def test_without_a_console_an_error_is_kept_in_a_file_and_shown(self):
        # the windowed exe has no console: the message must reach the user another way
        with mock.patch.object(start_schedule, "NO_CONSOLE", True), mock.patch.object(start_schedule, "HERE", self.root), \
                mock.patch("tkinter.messagebox.showerror") as box, mock.patch("tkinter.Tk"), contextlib.redirect_stderr(io.StringIO()):
            rc = start_schedule.main(["--config", str(self.root / "none.txt"), "--no-ui"])
        self.assertEqual(1, rc)
        self.assertIn("設定ファイルが見つかりません", (self.root / "plan_assist_error.txt").read_text(encoding="utf-8"))
        self.assertIn("設定ファイルが見つかりません", box.call_args[0][1])

    def test_with_a_console_nothing_is_written_and_no_box_is_shown(self):
        with mock.patch.object(start_schedule, "HERE", self.root), mock.patch("tkinter.messagebox.showerror") as box, \
                contextlib.redirect_stderr(io.StringIO()):
            start_schedule.main(["--config", str(self.root / "none.txt"), "--no-ui"])
        box.assert_not_called()
        self.assertFalse((self.root / "plan_assist_error.txt").exists())

    def test_a_crash_of_the_window_is_reported_not_raised(self):
        fake = types.ModuleType("start_window")
        fake.run_window = mock.Mock(side_effect=RuntimeError("window broke"))
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"start_window": fake}), contextlib.redirect_stderr(err):
            rc = start_schedule.main(["--config", str(self.root / "plan_assist.config.txt")])
        self.assertEqual(1, rc)
        self.assertIn("予期しないエラー", err.getvalue())
        self.assertIn("window broke", err.getvalue())

    def test_the_date_formats(self):
        self.assertEqual(date(2026, 11, 17), start_schedule.parse_day("20261117"))
        self.assertEqual(date(2026, 11, 17), start_schedule.parse_day("2026/11/17"))
        self.assertEqual(date(2026, 11, 7), start_schedule.parse_day("2026-11-07"))


if __name__ == "__main__":
    unittest.main()
