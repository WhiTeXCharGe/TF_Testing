"""The window of 01_StartSchedule, driven from the test (skipped when there is no screen). Message boxes are replaced."""
import sys
import tempfile
import time
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_tools import make_tools   # noqa: E402
from pa import weeks   # noqa: E402
from pa.settings import Settings   # noqa: E402

try:
    import tkinter
    from pa import ui
    tkinter.Tk().destroy()
    HAS_SCREEN = True
except Exception:          # no tkinter or no display
    HAS_SCREEN = False


@unittest.skipUnless(HAS_SCREEN, "no screen for tkinter")
class Window(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tools = make_tools(self.root / "Weekly_planning")
        self.su, self.sb = self.root / "su.xlsm", self.root / "list.xlsm"
        self.su.write_bytes(b"1")
        self.sb.write_bytes(b"2")
        self.settings = Settings({"mode": "local"}, base_dir=self.root)
        self.window = ui.StartWindow(self.settings, date(2026, 11, 17))
        self.addCleanup(self.window.root.destroy)

    def pump(self, until, seconds=10):
        end = time.time() + seconds
        while time.time() < end and not until():
            self.window.root.update()
            time.sleep(0.01)
        self.window.root.update()

    def text(self):
        return self.window.log.get("1.0", "end")

    def test_without_both_files_it_asks_and_runs_nothing(self):
        with mock.patch.object(ui.messagebox, "showwarning") as warn:
            self.window.su_var.set(str(self.su))
            self.window._start()
        warn.assert_called_once()
        self.assertFalse((self.root / "Work").exists())

    def test_start_runs_the_pipeline_in_the_background_and_shows_the_log_and_the_result(self):
        self.window.su_var.set(str(self.su))
        self.window.sb_var.set(str(self.sb))
        with mock.patch.object(ui.messagebox, "showinfo") as info, mock.patch.object(ui.messagebox, "showerror") as error:
            self.window._start()
            self.pump(lambda: info.called or error.called)
        info.assert_called_once()
        error.assert_not_called()
        self.assertIn("run-test", info.call_args[0][1])
        self.assertIn("== [4/4] 2-6", self.text())
        self.assertEqual("submitted", weeks.read_status(self.root / "Work" / "20261117")["stage"])
        self.assertEqual("finished", self.window.status.cget("text"))

    def test_the_submit_box_switches_the_last_step_off(self):
        self.window.su_var.set(str(self.su))
        self.window.sb_var.set(str(self.sb))
        self.window.submit_var.set(False)
        with mock.patch.object(ui.messagebox, "showinfo") as info:
            self.window._start()
            self.pump(lambda: info.called)
        self.assertEqual("ready", weeks.read_status(self.root / "Work" / "20261117")["stage"])

    def test_a_failure_is_shown_as_an_error_and_the_button_works_again(self):
        (self.tools / "fail_2-4").write_text("Error: boom", encoding="utf-8")
        self.window.su_var.set(str(self.su))
        self.window.sb_var.set(str(self.sb))
        with mock.patch.object(ui.messagebox, "showerror") as error:
            self.window._start()
            self.pump(lambda: error.called)
        error.assert_called_once()
        self.assertIn("boom", error.call_args[0][1])
        self.assertIn("2-4", error.call_args[0][1])
        self.assertNotIn("disabled", self.window.start_button.state())

    def test_a_week_that_was_sent_asks_before_running_again(self):
        self.window.su_var.set(str(self.su))
        self.window.sb_var.set(str(self.sb))
        with mock.patch.object(ui.messagebox, "showinfo") as info:
            self.window._start()
            self.pump(lambda: info.called)
        calls_before = (self.tools / "calls.json").read_text(encoding="utf-8")
        with mock.patch.object(ui.messagebox, "askyesno", return_value=False) as ask:
            self.window._start()
        ask.assert_called_once()
        self.assertEqual(calls_before, (self.tools / "calls.json").read_text(encoding="utf-8"))      # said no: nothing ran
        with mock.patch.object(ui.messagebox, "askyesno", return_value=True), mock.patch.object(ui.messagebox, "showinfo") as info:
            self.window._start()
            self.pump(lambda: info.called)
        self.assertTrue(info.called)
        self.assertTrue([p for p in (self.root / "Work" / "20261117").iterdir() if p.name.startswith("steps_old_")])


if __name__ == "__main__":
    unittest.main()
