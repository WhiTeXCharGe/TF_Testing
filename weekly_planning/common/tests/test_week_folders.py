"""week_folders: Work/<YYYYMMDD>/ of the programs the weekly user runs."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
from common import week_folders as weeks   # noqa: E402


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


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


if __name__ == "__main__":
    unittest.main()
