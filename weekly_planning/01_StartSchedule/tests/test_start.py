"""run_start with fake tools: the folders, the previous week, the arguments of every tool, the status and the failures."""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # weekly_planning/ (the folder of common)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # 01_StartSchedule/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import week_folders as weeks   # noqa: E402
from common.file_source import FileSource   # noqa: E402
from common.plan_settings import Settings   # noqa: E402
from fake_tools import calls, make_tools   # noqa: E402
from start_run import run_start   # noqa: E402

DAY = date(2026, 11, 17)
WEEK = "20261117"


class Base(unittest.TestCase):
    values = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.tools = make_tools(self.root / "Weekly_planning")
        (self.root / "Master").mkdir()
        self.su = self.root / "in" / "20261117 SU_Others.xlsm"
        self.sb = self.root / "in" / "製番リスト.xlsm"
        self.su.parent.mkdir()
        self.su.write_bytes(b"su others")
        self.sb.write_bytes(b"seiban list")
        self.log = []

    def settings(self, **over):
        values = {"mode": "local", "submit": "true"}
        values.update(self.values)
        values.update(over)
        return Settings(values, base_dir=self.root)

    def run_start(self, settings=None, **kw):
        return run_start(settings or self.settings(), kw.pop("su", self.su), kw.pop("sb", self.sb), kw.pop("day", DAY),
                         log=self.log.append, **kw)

    @property
    def week(self):
        return self.root / "Work" / WEEK

    def argv(self, step):
        return next(c["argv"] for c in calls(self.tools) if c["step"] == step)

    @staticmethod
    def value(argv, name):
        return argv[argv.index(name) + 1]

    def make_week(self, name, completed=True):
        folder = self.root / "Work" / name
        folder.mkdir(parents=True)
        if completed:
            (folder / "FlexibleSchedule.yaml").write_text("flex\n", encoding="utf-8")
            (folder / "EnvConfig.yaml").write_text("env\n", encoding="utf-8")
        return folder


class FirstWeek(Base):
    def test_it_runs_the_four_tools_in_order_and_reports_the_run(self):
        result = self.run_start()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(["2-2-3", "2-4", "2-5", "2-6"], [c["step"] for c in calls(self.tools)])
        self.assertEqual(("done", "submitted", "run-test"), (result.step, result.stage, result.run_id))
        self.assertIn("run-test", result.message)
        status = weeks.read_status(self.week)
        self.assertEqual(("submitted", "done", "run-test", True), (status["stage"], status["step"], status["run_id"], status["first_week"]))

    def test_the_two_files_are_copied_with_fixed_names_and_the_originals_stay(self):
        self.run_start()
        self.assertEqual(b"su others", (self.week / "input" / "01_SU_Others.xlsm").read_bytes())
        self.assertEqual(b"seiban list", (self.week / "input" / "02_SeibanList.xlsm").read_bytes())
        self.assertEqual(b"su others", self.su.read_bytes())
        sb = self.root / "in" / "list.xlsx"
        sb.write_bytes(b"x")
        self.run_start(sb=sb, overwrite=True)
        self.assertTrue((self.week / "input" / "02_SeibanList.xlsx").exists())     # the extension of the user's file is kept

    def test_without_master_files_the_week_starts_empty_with_two_warnings(self):
        self.run_start()
        argv = self.argv("2-2-3")
        self.assertEqual("{}\n", Path(self.value(argv, "--prev-flexible")).read_text(encoding="utf-8"))
        self.assertEqual("", Path(self.value(argv, "--prev-env")).read_text(encoding="utf-8"))
        self.assertNotIn("--skill-file", argv)
        text = "\n".join(self.log)
        self.assertIn("警告: MasterフォルダにEnvConfig_start.yamlがありません", text)
        self.assertIn("警告: Masterフォルダにスキルレベルのファイルがありません", text)

    def test_the_master_files_are_used_in_the_first_week(self):
        (self.root / "Master" / "EnvConfig_start.yaml").write_text("start env\n", encoding="utf-8")
        (self.root / "Master" / "20260101 SU_Others_skill level.xlsm").write_bytes(b"old skills")
        (self.root / "Master" / "20260915 SU_Others_skill level.xlsm").write_bytes(b"new skills")
        (self.root / "Master" / "~$20260915 SU_Others_skill level.xlsm").write_bytes(b"lock file")
        self.run_start()
        argv = self.argv("2-2-3")
        self.assertEqual("start env\n", Path(self.value(argv, "--prev-env")).read_text(encoding="utf-8"))
        self.assertEqual("20260915 SU_Others_skill level.xlsm", Path(self.value(argv, "--skill-file")).name)   # the newest one

    def test_the_arguments_of_2_2_3(self):
        self.run_start()
        argv = self.argv("2-2-3")
        self.assertEqual(str(self.week / "input" / "01_SU_Others.xlsm"), self.value(argv, "--su-others"))
        self.assertEqual(str(self.week / "input" / "02_SeibanList.xlsm"), self.value(argv, "--seiban-list"))
        self.assertEqual(str(self.week / "steps"), self.value(argv, "--out-dir"))
        self.assertEqual("2026/11/17", self.value(argv, "--today"))
        self.assertNotIn("--report", argv)        # no report files are written in the week folder

    def test_sheet_names_and_the_default_level_are_passed_only_when_they_are_set(self):
        self.run_start()
        argv = self.argv("2-2-3")
        for option in ("--su-sheets", "--skill-sheet", "--seiban-sheet", "--default-skill-level"):
            self.assertNotIn(option, argv)
        self.run_start(self.settings(su_sheet="予定表_2027", skill_sheet="予定表_2027 (2)", seiban_sheet="製番", default_skill_level="2"),
                       overwrite=True)
        argv = [c["argv"] for c in calls(self.tools) if c["step"] == "2-2-3"][-1]
        self.assertEqual("予定表_2027", self.value(argv, "--su-sheets"))
        self.assertEqual("予定表_2027 (2)", self.value(argv, "--skill-sheet"))
        self.assertEqual("製番", self.value(argv, "--seiban-sheet"))
        self.assertEqual("2", self.value(argv, "--default-skill-level"))

    def test_2_4_and_2_6_get_their_own_settings_file_in_steps_and_2_5_gets_the_02_and_03_files(self):
        self.run_start(self.settings(stage1_minutes="45", stage2_enabled="true", run_id="", azure_url="https://x.example"))
        steps = self.week / "steps"
        self.assertEqual(["--config", str(steps / "config_2-4.txt")], self.argv("2-4"))
        cfg = (steps / "config_2-4.txt").read_text(encoding="utf-8")
        self.assertIn(f"schedule = {steps / '01_Schedule.yaml'}", cfg)
        self.assertIn(f"seiban_list = {self.week / 'input' / '02_SeibanList.xlsm'}", cfg)
        self.assertIn("report = false", cfg)
        a5 = self.argv("2-5")
        self.assertEqual(str(steps / "02_Schedule.yaml"), self.value(a5, "--input-schedule"))
        self.assertEqual(str(steps / "03_EnvConfig.yaml"), self.value(a5, "--output-env-config"))
        self.assertEqual(["--config", str(steps / "config_2-6.txt")], self.argv("2-6"))
        cfg6 = (steps / "config_2-6.txt").read_text(encoding="utf-8")
        self.assertIn(f"schedule = {steps / '03_Schedule.yaml'}", cfg6)
        self.assertIn(f"out_dir = {steps}", cfg6)
        self.assertIn("mode = local", cfg6)
        self.assertIn("stage1_minutes = 45", cfg6)
        self.assertIn("stage2_enabled = true", cfg6)
        self.assertNotIn("run_id", cfg6)                  # an empty setting is left out: the tool makes a new name
        self.assertNotIn("random_seed", cfg6)

    def test_the_mode_is_online_when_nothing_is_set(self):
        self.run_start(self.settings(mode="", azure_url="https://x.example"))
        self.assertIn("mode = online", (self.week / "steps" / "config_2-6.txt").read_text(encoding="utf-8"))

    def test_the_log_file_holds_every_line(self):
        self.run_start()
        text = (self.week / "logs" / "start_log.txt").read_text(encoding="utf-8")
        self.assertIn("== [1/4] 2-2-3", text)
        self.assertIn("== [4/4] 2-6", text)
        self.assertIn("reading SU_Others", text)
        self.assertEqual(len(self.log), len(text.splitlines()))

    def test_unknown_settings_are_mentioned(self):
        s = self.settings()
        s.unknown = ["stage1_minuts"]
        self.run_start(s)
        self.assertTrue(any("項目 'stage1_minuts' は不明" in line for line in self.log))


class LaterWeeks(Base):
    def test_the_previous_week_is_the_newest_completed_week_before_this_one(self):
        self.make_week("20261103")
        self.make_week("20261110")
        self.make_week("20261114", completed=False)            # started but never finished
        self.make_week("20261124")                              # later than this week
        (self.root / "Master" / "20260915 SU_Others_skill level.xlsm").write_bytes(b"skills")
        self.run_start()
        argv = self.argv("2-2-3")
        self.assertEqual(str(self.root / "Work" / "20261110" / "FlexibleSchedule.yaml"), self.value(argv, "--prev-flexible"))
        self.assertEqual(str(self.root / "Work" / "20261110" / "EnvConfig.yaml"), self.value(argv, "--prev-env"))
        self.assertNotIn("--skill-file", argv)                  # the skill workbook is for the first week only
        status = weeks.read_status(self.week)
        self.assertEqual(("20261110", False), (status["previous_week"], status["first_week"]))
        self.assertIn("前週: 20261110", self.log)
        self.assertFalse(any("初回の週です" in line for line in self.log))


class NoSubmit(Base):
    def test_it_stops_after_2_5_and_the_week_is_ready(self):
        result = self.run_start(submit=False)
        self.assertTrue(result.ok)
        self.assertEqual("ready", result.stage)
        self.assertEqual(["2-2-3", "2-4", "2-5"], [c["step"] for c in calls(self.tools)])
        self.assertIn("== [3/3] 2-5", "\n".join(self.log))
        self.assertFalse((self.week / "steps" / "run_info.json").exists())
        self.assertEqual("ready", weeks.read_status(self.week)["stage"])

    def test_the_setting_submit_false_does_the_same_and_no_address_is_needed(self):
        result = self.run_start(self.settings(submit="false", mode="online", azure_url="https://<your-solver-api>.example"))
        self.assertTrue(result.ok, result.message)
        self.assertEqual("ready", result.stage)


class Failures(Base):
    def test_a_tool_that_fails_stops_the_run_and_the_week_is_marked_failed(self):
        (self.tools / "fail_2-4").write_text("Error: seiban list has no sheet", encoding="utf-8")
        result = self.run_start()
        self.assertFalse(result.ok)
        self.assertEqual(("2-4", "failed"), (result.step, result.stage))
        self.assertIn("seiban list has no sheet", result.message)
        self.assertEqual(["2-2-3", "2-4"], [c["step"] for c in calls(self.tools)])     # 2-5 and 2-6 never ran
        status = weeks.read_status(self.week)
        self.assertEqual(("failed", "2-4"), (status["stage"], status["step"]))
        self.assertIn("中断（2-4 製番リストの反映）", (self.week / "logs" / "start_log.txt").read_text(encoding="utf-8"))

    def test_a_tool_that_raises_is_caught(self):
        (self.tools / "raise_2-5").write_text("", encoding="utf-8")
        result = self.run_start()
        self.assertEqual(("2-5", False), (result.step, result.ok))
        self.assertIn("tool exploded", result.message)
        self.assertEqual("failed", weeks.read_status(self.week)["stage"])

    def test_a_tool_that_does_not_write_its_output_is_reported(self):
        (self.tools / "2-5set_plan_period" / "set_plan_period.py").write_text(
            "def main(argv):\n    return 0\n", encoding="utf-8")
        result = self.run_start()
        self.assertEqual("2-5", result.step)
        self.assertIn("03_Schedule.yaml が作られていません", result.message)

    def test_a_failed_week_can_be_run_again_and_the_first_try_is_kept(self):
        (self.tools / "fail_2-4").write_text("Error: boom", encoding="utf-8")
        self.run_start()
        (self.tools / "fail_2-4").unlink()
        result = self.run_start()
        self.assertTrue(result.ok)
        old = [p.name for p in self.week.iterdir() if "_old_" in p.name]
        self.assertEqual(2, len(old))                                 # steps_old_<time> and logs_old_<time>
        self.assertTrue((self.week / "steps" / "03_Schedule.yaml").exists())


class Rerun(Base):
    def test_a_week_that_was_sent_is_not_run_again_without_the_ok(self):
        self.run_start()
        before = len(calls(self.tools))
        result = self.run_start()
        self.assertFalse(result.ok)
        self.assertEqual("check", result.step)
        self.assertIn("送信済み", result.message)
        self.assertEqual(before, len(calls(self.tools)))
        self.assertEqual("submitted", weeks.read_status(self.week)["stage"])      # nothing was touched
        self.assertFalse([p for p in self.week.iterdir() if "_old_" in p.name])

    def test_with_the_ok_the_old_files_are_renamed_not_deleted(self):
        self.run_start()
        (self.week / "steps" / "mine.txt").write_text("keep", encoding="utf-8")
        result = self.run_start(overwrite=True)
        self.assertTrue(result.ok)
        kept = [p for p in self.week.iterdir() if p.name.startswith("steps_old_")]
        self.assertEqual(1, len(kept))
        self.assertEqual("keep", (kept[0] / "mine.txt").read_text(encoding="utf-8"))
        self.assertFalse((self.week / "steps" / "mine.txt").exists())

    def test_a_week_that_was_only_prepared_is_simply_run_again(self):
        self.assertEqual("ready", self.run_start(submit=False).stage)
        result = self.run_start(submit=True)
        self.assertTrue(result.ok)
        self.assertEqual("submitted", result.stage)


class FakeSharePoint:
    """The SharePoint client of a test: link -> (file name, content)."""

    def __init__(self, files):
        self.files, self.asked = files, []

    def get_file_url_from_sharing_link(self, link):
        self.asked.append(link)
        if link not in self.files:
            raise RuntimeError("HTTP 404: itemNotFound")
        return link

    def get_file_name(self, url):
        return self.files[url][0]

    def download_file(self, url, destination):
        Path(destination).write_bytes(self.files[url][1])


class FromSharePoint(Base):
    SU, SB = "https://example.sharepoint.com/su", "https://example.sharepoint.com/sb"

    def setUp(self):
        super().setUp()
        self.client = FakeSharePoint({self.SU: ("SU_Others.xlsm", b"remote su"), self.SB: ("List.xlsx", b"remote list")})
        self.source = FileSource("tenant", "client", client_factory=lambda: self.client)

    def test_two_links_are_downloaded_into_the_week_and_the_tools_work_from_the_copies(self):
        result = self.run_start(su=self.SU, sb=self.SB, source=self.source)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(b"remote su", (self.week / "input" / "01_SU_Others.xlsm").read_bytes())
        self.assertEqual(b"remote list", (self.week / "input" / "02_SeibanList.xlsx").read_bytes())      # the extension of the original
        self.assertEqual(str(self.week / "input" / "01_SU_Others.xlsm"), self.value(self.argv("2-2-3"), "--su-others"))
        self.assertEqual(str(self.week / "input" / "02_SeibanList.xlsx"), self.value(self.argv("2-2-3"), "--seiban-list"))
        text = " | ".join(self.log)
        self.assertIn("SU_OthersをSharePointから取得しています", text)
        self.assertIn("SU_Others: SU_Others.xlsm", text)
        self.assertIn("製番リスト: List.xlsx", text)
        self.assertEqual([self.SU, self.SB], self.client.asked)

    def test_a_link_and_a_local_file_can_be_mixed(self):
        result = self.run_start(su=self.SU, sb=self.sb, source=self.source)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(b"seiban list", (self.week / "input" / "02_SeibanList.xlsm").read_bytes())
        self.assertEqual([self.SU], self.client.asked)

    def test_a_file_that_the_update_has_locked_stops_the_run_before_any_tool(self):
        self.client.files[self.SU] = ("_SU_Others.xlsx", b"x")
        result = self.run_start(su=self.SU, sb=self.SB, source=self.source)
        self.assertFalse(result.ok)
        self.assertEqual("download", result.step)
        self.assertIn("ロックされています", result.message)
        self.assertIn("SU_Others", result.message)
        self.assertEqual([], calls(self.tools))
        self.assertEqual("failed", weeks.read_status(self.week)["stage"])

    def test_a_link_that_cannot_be_opened_is_reported_with_the_reason(self):
        result = self.run_start(su=self.SU, sb="https://example.sharepoint.com/other", source=self.source)
        self.assertEqual("download", result.step)
        self.assertIn("製番リスト", result.message)
        self.assertIn("itemNotFound", result.message)

    def test_a_link_without_the_settings_stops_in_the_check_and_creates_nothing(self):
        result = self.run_start(su=self.SU, sb=self.SB)           # no tenant_id / client_id in the settings
        self.assertEqual("check", result.step)
        self.assertIn("tenant_id と client_id", result.message)
        self.assertFalse((self.root / "Work").exists())
        self.assertEqual([], calls(self.tools))

    def test_the_ids_of_the_settings_reach_the_file_source(self):
        made = {}

        class Spy(FileSource):
            def __init__(spy, tenant_id=None, client_id=None, sign_in=None, client_factory=None):
                made.update(tenant=tenant_id, client=client_id, sign_in=sign_in)
                super().__init__(tenant_id, client_id, sign_in, lambda: self.client)

        sign_in = lambda url, code: True      # noqa: E731
        from unittest import mock
        with mock.patch("start_run.FileSource", Spy):
            result = self.run_start(settings=self.settings(tenant_id="t-1", client_id="c-1"), su=self.SU, sb=self.SB, sign_in=sign_in)
        self.assertTrue(result.ok, result.message)
        self.assertEqual({"tenant": "t-1", "client": "c-1", "sign_in": sign_in}, made)

    def test_the_same_link_twice_is_refused(self):
        result = self.run_start(su=self.SU, sb=self.SU, source=self.source)
        self.assertEqual("check", result.step)
        self.assertIn("同じファイル", result.message)


class Checks(Base):
    def stops_in_check(self, expected, **kw):
        result = self.run_start(**kw)
        self.assertFalse(result.ok)
        self.assertEqual("check", result.step)
        self.assertIn(expected, result.message)
        self.assertEqual([], calls(self.tools))
        self.assertFalse((self.root / "Work").exists())             # nothing was created
        return result

    def test_a_missing_file(self):
        self.stops_in_check("SU_Othersのファイルが見つかりません", su=self.root / "nope.xlsm")
        self.stops_in_check("製番リストのファイルが選択されていません", sb="")

    def test_a_file_that_is_not_excel(self):
        csv = self.root / "in" / "list.csv"
        csv.write_text("x", encoding="utf-8")
        self.stops_in_check("Excelファイル", sb=csv)

    def test_the_same_file_twice(self):
        self.stops_in_check("同じファイル", sb=self.su)

    def test_incomplete_scripts(self):
        (self.tools / "2-4apply_seiban_list" / "apply_seiban_list.py").unlink()
        self.stops_in_check("apply_seiban_list.py", )

    def test_a_missing_script_of_2_6_does_not_matter_without_submit(self):
        (self.tools / "2-6submit_to_timefold" / "submit_to_timefold.py").unlink()
        self.assertTrue(self.run_start(submit=False).ok)

    def test_the_solver_address_is_checked_before_anything_runs(self):
        self.stops_in_check("azure_url", settings=self.settings(mode="online", azure_url="https://<your-solver-api>.example"))
        self.stops_in_check("azure_url", settings=self.settings(mode="", azure_url=""))
        self.stops_in_check("mode は online か local", settings=self.settings(mode="cloud"))


if __name__ == "__main__":
    unittest.main()
