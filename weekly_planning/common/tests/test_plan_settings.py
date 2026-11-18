"""plan_settings: the settings file of the programs the weekly user runs."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common import plan_settings as st   # noqa: E402


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
        s = st.load(program_dir=self.root)                      # no settings file next to the program: defaults
        self.assertEqual(({}, self.root), (s.values, s.base_dir))
        (self.root / st.CONFIG_NAME).write_text("mode = local\n", encoding="utf-8")
        self.assertEqual("local", st.load(program_dir=self.root).text("mode"))
        with self.assertRaises(st.SettingsError):
            st.load(self.root / "nope.txt")

    def test_the_tools_folder_is_found_next_to_the_program_or_in_the_repository_layout(self):
        base = self.root / "01_StartSchedule"
        base.mkdir()
        (self.root / "common").mkdir()                                    # the program folder is inside weekly_planning
        self.assertEqual(self.root, st.Settings(base_dir=base).tools_dir)
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


if __name__ == "__main__":
    unittest.main()
