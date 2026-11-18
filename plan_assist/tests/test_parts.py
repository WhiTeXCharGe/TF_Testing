"""settings, week folders and the tool runner of plan_assist, and the command line of start_schedule.py."""
import contextlib
import io
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import start_schedule   # noqa: E402
from fake_tools import calls, make_tools   # noqa: E402
from pa import settings as st, weeks   # noqa: E402
from pa.tools import ToolRunner   # noqa: E402


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class SettingsFile(TempCase):
    def write(self, text):
        p = self.root / "plan_assist.config.txt"
        p.write_text(text, encoding="utf-8")
        return p

    def test_keys_comments_quotes_and_empty_values(self):
        s = st.load(self.write("# a comment\nMode = online\nazure-url = 'https://x.example'  \nstage1_minutes = 45   # minutes\n"
                               "run_id =\n; also a comment\nsu_sheet = 予定表_2026\n"))
        self.assertEqual("online", s.text("mode"))
        self.assertEqual("https://x.example", s.text("azure_url"))
        self.assertEqual(45, s.whole("stage1_minutes"))
        self.assertFalse(s.has("run_id"))
        self.assertEqual("予定表_2026", s.text("su_sheet"))
        self.assertEqual([], s.unknown)

    def test_unknown_keys_are_listed_once(self):
        s = st.load(self.write("stage1_minuts = 3\nstage1_minuts = 4\nmode = local\n"))
        self.assertEqual(["stage1_minuts"], s.unknown)

    def test_relative_paths_are_relative_to_the_settings_file_and_defaults_are_next_to_it(self):
        absolute = self.root / "elsewhere" / "master"
        s = st.load(self.write(f"work_dir = sub/work\nmaster_dir = {absolute}\n"))
        self.assertEqual(self.root / "sub/work", s.work_dir)
        self.assertEqual(absolute, s.master_dir)
        self.assertEqual(self.root / "Work", st.Settings(base_dir=self.root).work_dir)
        self.assertEqual(self.root / "Master", st.Settings(base_dir=self.root).master_dir)

    def test_flags_and_numbers_are_checked(self):
        s = st.load(self.write("submit = maybe\nstage1_minutes = soon\n"))
        with self.assertRaises(st.SettingsError):
            s.flag("submit", True)
        with self.assertRaises(st.SettingsError):
            s.whole("stage1_minutes")
        self.assertTrue(st.load(self.write("submit = Yes\n")).flag("submit", False))
        self.assertTrue(st.Settings(base_dir=self.root).flag("submit", True))

    def test_a_missing_default_file_gives_defaults_but_a_missing_asked_for_file_is_an_error(self):
        with mock.patch.object(st, "app_dir", return_value=self.root):          # no settings file next to the program
            s = st.load()
            self.assertEqual(({}, self.root), (s.values, s.base_dir))
        with self.assertRaises(st.SettingsError):
            st.load(self.root / "nope.txt")

    def test_the_tools_folder_is_found_next_to_the_program_or_in_the_repository_layout(self):
        base = self.root / "plan_assist"
        base.mkdir()
        (self.root / "weekly_planning" / "common").mkdir(parents=True)
        self.assertEqual(self.root / "weekly_planning", st.Settings(base_dir=base).tools_dir)
        (base / "Weekly_planning" / "common").mkdir(parents=True)
        self.assertEqual(base / "Weekly_planning", st.Settings(base_dir=base).tools_dir)
        self.assertEqual(self.root / "else", st.Settings({"tools_dir": str(self.root / "else")}, base_dir=base).tools_dir)

    def test_master_files(self):
        s = st.Settings(base_dir=self.root)
        self.assertIsNone(s.master_skill_file())
        self.assertIsNone(s.master_start_env())
        (self.root / "Master").mkdir()
        (self.root / "Master" / "EnvConfig_start.yaml").write_text("x", encoding="utf-8")
        (self.root / "Master" / "a skill 1.xlsm").write_bytes(b"1")
        (self.root / "Master" / "b skill 2.xlsm").write_bytes(b"2")
        self.assertEqual("EnvConfig_start.yaml", s.master_start_env().name)
        self.assertEqual("b skill 2.xlsm", s.master_skill_file().name)
        explicit = st.Settings({"skill_file": "Master/a skill 1.xlsm", "start_env": "other.yaml"}, base_dir=self.root)
        self.assertEqual("a skill 1.xlsm", explicit.master_skill_file().name)
        self.assertIsNone(explicit.master_start_env())
        self.assertIsNone(st.Settings({"skill_file": "nope.xlsm"}, base_dir=self.root).master_skill_file())


class WeekFolders(TempCase):
    def make(self, name, files=weeks.KEY_FILES, empty=()):
        folder = self.root / name
        folder.mkdir(parents=True)
        for f in files:
            (folder / f).write_text("" if f in empty else "x", encoding="utf-8")
        return folder

    def test_a_week_is_completed_only_with_both_non_empty_key_files(self):
        self.assertTrue(weeks.is_completed(self.make("20261110")))
        self.assertFalse(weeks.is_completed(self.make("20261111", files=("FlexibleSchedule.yaml",))))
        self.assertFalse(weeks.is_completed(self.make("20261112", empty=("EnvConfig.yaml",))))

    def test_the_previous_week_is_the_newest_completed_one_before_the_name(self):
        for n in ("20261101", "20261108", "20261115", "20261122"):
            self.make(n)
        self.make("20261112", files=())                        # not completed
        self.make("20261109x")                                 # completed, but not a date folder
        (self.root / "notes").mkdir()
        (self.root / "20261109").write_text("a file, not a folder", encoding="utf-8")
        self.assertEqual("20261108", weeks.previous_week(self.root, "20261110").name)
        self.assertEqual("20261115", weeks.previous_week(self.root, "20261122").name)
        self.assertEqual("20261108", weeks.previous_week(self.root, "20261115").name)
        self.assertIsNone(weeks.previous_week(self.root, "20261101"))
        self.assertIsNone(weeks.previous_week(self.root / "nowhere", "20261110"))

    def test_status_is_merged_and_stamped_and_a_broken_file_reads_as_empty(self):
        folder = self.root / "20261117"
        weeks.write_status(folder, stage="started", step="check")
        status = weeks.write_status(folder, step="2-4")
        self.assertEqual(("started", "2-4"), (status["stage"], status["step"]))
        self.assertIn("updated", status)
        (folder / weeks.STATUS_FILE).write_text("{not json", encoding="utf-8")
        self.assertEqual({}, weeks.read_status(folder))
        self.assertEqual({}, weeks.read_status(self.root / "nowhere"))

    def test_was_submitted_by_the_run_record_or_the_status(self):
        folder = self.root / "20261117"
        self.assertFalse(weeks.was_submitted(folder))
        weeks.write_status(folder, stage="ready")
        self.assertFalse(weeks.was_submitted(folder))
        weeks.write_status(folder, stage="submitted")
        self.assertTrue(weeks.was_submitted(folder))
        weeks.write_status(folder, stage="failed")
        self.assertFalse(weeks.was_submitted(folder))
        (folder / "steps").mkdir()
        (folder / "steps" / weeks.RUN_INFO_FILE).write_text("{}", encoding="utf-8")
        self.assertTrue(weeks.was_submitted(folder))

    def test_archive_renames_steps_and_logs_and_leaves_the_rest(self):
        folder = self.root / "20261117"
        (folder / "steps").mkdir(parents=True)
        (folder / "logs").mkdir()
        (folder / "input").mkdir()
        (folder / "steps" / "a.txt").write_text("a", encoding="utf-8")
        self.assertEqual(["steps_old_T1", "logs_old_T1"], weeks.archive_old(folder, "T1"))
        self.assertTrue((folder / "steps_old_T1" / "a.txt").exists())
        self.assertTrue((folder / "input").exists())
        self.assertEqual([], weeks.archive_old(folder, "T2"))


class Runner(TempCase):
    def runner_with(self, source):
        tools = make_tools(self.root / "tools")
        (tools / "2-5set_plan_period" / "set_plan_period.py").write_text(source, encoding="utf-8")
        return ToolRunner(tools)

    def test_the_output_arrives_line_by_line_and_the_exit_code_is_returned(self):
        runner = self.runner_with("import sys\ndef main(argv):\n    print('a')\n    print('b', 'c')\n"
                                  "    print('oops', file=sys.stderr)\n    sys.stdout.write('no newline')\n    return 3\n")
        seen = []
        result = runner.run("2-5", ["--x"], seen.append)
        self.assertEqual(3, result.rc)
        self.assertEqual(["a", "b c", "oops", "no newline"], seen)
        self.assertEqual(["oops"], result.errors)
        self.assertEqual("oops", result.last_error())

    def test_none_is_success_and_exit_and_exceptions_are_caught(self):
        self.assertEqual(0, self.runner_with("def main(argv):\n    pass\n").run("2-5", []).rc)
        self.assertEqual(2, self.runner_with("import sys\ndef main(argv):\n    sys.exit(2)\n").run("2-5", []).rc)
        self.assertEqual(0, self.runner_with("import sys\ndef main(argv):\n    sys.exit()\n").run("2-5", []).rc)
        self.assertEqual(1, self.runner_with("import sys\ndef main(argv):\n    sys.exit('bad')\n").run("2-5", []).rc)
        result = self.runner_with("def main(argv):\n    raise ValueError('wrong value')\n").run("2-5", [])
        self.assertEqual(1, result.rc)
        self.assertEqual("ValueError: wrong value", result.last_error())

    def test_argparse_errors_do_not_stop_the_program(self):
        runner = self.runner_with("import argparse\ndef main(argv):\n    ap = argparse.ArgumentParser()\n"
                                  "    ap.add_argument('--needed', required=True)\n    ap.parse_args(argv)\n")
        result = runner.run("2-5", [])
        self.assertEqual(2, result.rc)
        self.assertIn("--needed", result.last_error())

    def test_arguments_are_given_as_text_and_warnings_are_not_printed(self):
        runner = self.runner_with("import warnings\ndef main(argv):\n    warnings.warn('noise')\n    print(argv)\n")
        seen = []
        runner.run("2-5", [Path("a"), 3], seen.append)
        self.assertEqual(["['a', '3']"], seen)

    def test_missing_scripts_are_listed(self):
        runner = ToolRunner(self.root / "empty")
        self.assertEqual(2, len(runner.missing(["2-2-3"])))               # common and the script
        make_tools(self.root / "empty")
        self.assertEqual([], runner.missing(["2-2-3", "2-4", "2-5", "2-6"]))
        self.assertEqual(1, len(runner.missing(["3-1"])))


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
        self.assertIn("Week 20261117", text)
        self.assertIn("== [4/4] 2-6", text)
        self.assertEqual(["2-2-3", "2-4", "2-5", "2-6"], [c["step"] for c in calls(self.tools)])

    def test_no_submit_and_the_exit_code_of_a_failure(self):
        rc, text = self.run_main(self.args("--no-submit"))
        self.assertEqual(0, rc)
        self.assertNotIn("2-6", [c["step"] for c in calls(self.tools)])
        (self.tools / "fail_2-4").write_text("Error: boom", encoding="utf-8")
        rc, text = self.run_main(self.args("--no-submit", "--yes"))
        self.assertEqual(1, rc)
        self.assertIn("STOPPED at 2-4", text)

    def test_a_sent_week_needs_yes(self):
        self.assertEqual(0, self.run_main(self.args())[0])
        rc, text = self.run_main(self.args())
        self.assertEqual(1, rc)
        self.assertIn("already sent", text)
        self.assertEqual(0, self.run_main(self.args("--yes"))[0])

    def test_no_ui_without_files_is_an_error_and_a_bad_date_is_refused(self):
        rc, text = self.run_main(["--config", str(self.root / "plan_assist.config.txt"), "--no-ui"])
        self.assertEqual(1, rc)
        self.assertIn("not chosen", text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            start_schedule.main(self.args()[:-2] + ["--date", "soon"])

    def test_a_missing_settings_file_is_reported(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = start_schedule.main(["--config", str(self.root / "none.txt"), "--no-ui"])
        self.assertEqual(1, rc)
        self.assertIn("settings file not found", err.getvalue())

    def test_the_date_formats(self):
        self.assertEqual(date(2026, 11, 17), start_schedule.parse_day("20261117"))
        self.assertEqual(date(2026, 11, 17), start_schedule.parse_day("2026/11/17"))
        self.assertEqual(date(2026, 11, 7), start_schedule.parse_day("2026-11-07"))


if __name__ == "__main__":
    unittest.main()
