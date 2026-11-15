"""Tests of apply_seiban_list.py: the whole step on plain data, and the command line with config.txt (generated workbooks only).
Run from the tool folder:  python -m unittest discover -s tests -v"""
import contextlib
import copy
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import yaml
from openpyxl import Workbook
from openpyxl.styles import PatternFill

from envs import TEMPLATE_LIKE_WF_TOOL, TEMPLATE_WITH_P1, make_env, make_schedule, plan_row, seiban_data
import apply_seiban_list as app
from common import seiban_list as sl, tf_config as tfc


def run(rows, schedule=None, env=None, **kw):
    schedule = schedule or make_schedule()
    env = {"environment": make_env()} if env is None else env
    before = copy.deepcopy((schedule, env))
    out = app.apply_seiban_list(schedule, env, seiban_data(rows, **kw))
    assert (schedule, env) == before, "the inputs must not be changed"
    return out


def codes(out):
    return [t["name"] for t in out["schedule"]["schedule"]["workflow_task_list"]]


def lines(out, start):
    return [x for x in out["report"] if x.startswith(start)]


class Adding(unittest.TestCase):
    def test_a_white_row_becomes_a_module_and_the_masters_exist(self):
        out = run([plan_row(code="530N00005A", customer="Cust C", fab="Fab C", region="Region X", template=2)])
        env, sch = out["env"]["environment"], out["schedule"]["schedule"]
        task = sch["workflow_task_list"][-1]
        self.assertEqual(("e3", "530N00005A", "wf1"), (task["id"], task["name"], task["workflow"]))
        self.assertEqual("Fab C", next(f for f in env["fab_list"] if f["id"] == task["fab"])["name"])
        self.assertEqual("wf1", env["workflow_list"][-1]["id"])
        self.assertEqual({"added": 1, "deleted": 0, "skipped": 0, "kept": 0, "nothing_to_delete": 0, "customers": 1, "fabs": 1,
                          "regions": 0, "workflows": 1, "warnings": 0}, out["stats"])

    def test_a_template_that_matches_an_existing_workflow_creates_no_workflow(self):
        out = run([plan_row(template=1)])
        self.assertEqual(("wf_tool", 0), (out["schedule"]["schedule"]["workflow_task_list"][-1]["workflow"], out["stats"]["workflows"]))

    def test_rows_of_one_template_share_the_workflow_and_the_skills_are_added_once(self):
        out = run([plan_row(2, code="530N00005A", template=2), plan_row(3, code="530N00006A", template=2)])
        self.assertEqual(["wf1", "wf1"], [t["workflow"] for t in out["schedule"]["schedule"]["workflow_task_list"][-2:]])
        self.assertEqual(1, out["stats"]["workflows"])
        self.assertEqual(1, len([x for x in out["report"] if "template 2" in x]))              # the template is looked up once, not per row
        self.assertEqual(1, len(lines(out, "INFO: template 2: new workflow")))
        self.assertEqual(4, len(lines(out, "INFO: operation")))
        self.assertEqual(["e3", "e4"], [t["id"] for t in out["schedule"]["schedule"]["workflow_task_list"][-2:]])

    def test_existing_tasks_and_assignments_are_kept(self):
        out = run([plan_row()])
        self.assertEqual(make_schedule()["schedule"]["assignment_list"], out["schedule"]["schedule"]["assignment_list"])
        self.assertEqual(["530N00001A", "530N00002A", "530N00005A"], codes(out))

    def test_a_white_row_that_is_already_in_the_schedule_is_skipped_with_a_warning(self):
        out = run([plan_row(code="530N00002A", customer="Cust Z", fab="Fab Z")])
        self.assertEqual(["530N00001A", "530N00002A"], codes(out))
        self.assertEqual(1, out["stats"]["skipped"])
        self.assertIn("already a workflow task in the Schedule (e2)", lines(out, "WARN")[0])
        self.assertEqual((0, 0), (out["stats"]["customers"], out["stats"]["fabs"]))        # a skipped row creates nothing

    def test_an_invalid_white_row_is_skipped_and_creates_nothing(self):
        bad = plan_row(code="530N00005A", customer="Cust C", fab="Fab C", template=2, phases={1: ("2026/12/01", None)})
        out = run([bad])
        self.assertEqual(1, out["stats"]["skipped"])
        self.assertIn("row 2 (530N00005A): skipped: process 1 has no end date", lines(out, "WARN")[0])
        self.assertEqual((0, 0, 0), (out["stats"]["customers"], out["stats"]["fabs"], out["stats"]["workflows"]))
        self.assertEqual(make_env(), out["env"]["environment"])

    def test_a_row_without_a_template_sheet_is_skipped(self):
        out = run([plan_row(template=9)])
        self.assertIn("no sheet for template 9", lines(out, "WARN")[0])
        self.assertEqual(make_env(), out["env"]["environment"])

    def test_an_unknown_colour_is_skipped(self):
        row = plan_row(kind=sl.UNKNOWN, problems=["row colour FFFF00 is not white, grey or pink"])
        out = run([row])
        self.assertEqual((1, 0), (out["stats"]["skipped"], out["stats"]["added"]))
        self.assertIn("row colour FFFF00 is not white, grey or pink", lines(out, "WARN")[0])

    def test_the_same_seiban_twice_on_white_rows_is_added_once(self):
        out = run([plan_row(2, code="530N00005A"), plan_row(3, code="530N00005A")])
        self.assertEqual(["530N00001A", "530N00002A", "530N00005A"], codes(out))
        self.assertIn("row 3 (530N00005A): skipped: the same seiban is on an earlier white row", lines(out, "WARN")[0])

    def test_a_row_without_a_seiban_is_skipped(self):
        out = run([plan_row(code=None, problems=["seiban is empty"])])
        self.assertEqual(1, out["stats"]["skipped"])
        self.assertIn("skipped: seiban is empty", lines(out, "WARN")[0])


class Deleting(unittest.TestCase):
    def test_a_pink_row_removes_the_module_and_its_assignments(self):
        out = run([plan_row(kind=sl.DELETE, code="530N00001A")])
        sch = out["schedule"]["schedule"]
        self.assertEqual(["530N00002A"], codes(out))
        self.assertEqual(["e2p2o1", "misc_other_1"], [a["operation_task"] for a in sch["assignment_list"]])
        self.assertEqual(1, out["stats"]["deleted"])
        self.assertIn("module e1 deleted (3 phase tasks, 4 operation tasks, 3 assignment rows)", lines(out, "INFO")[0])

    def test_deleted_fixed_rows_are_a_warning(self):
        out = run([plan_row(kind=sl.DELETE, code="530N00001A")])
        self.assertEqual(["WARN: row 2 (530N00001A): 1 of the deleted assignment rows were Fixed"], lines(out, "WARN"))

    def test_a_pink_row_without_a_module_is_only_a_note(self):
        out = run([plan_row(kind=sl.DELETE, code="530N09999A")])
        self.assertEqual((0, 1, 0), (out["stats"]["deleted"], out["stats"]["nothing_to_delete"], out["stats"]["warnings"]))
        self.assertEqual(make_schedule(), out["schedule"])

    def test_a_pink_row_needs_no_template_and_no_dates(self):
        row = plan_row(kind=sl.DELETE, code="530N00001A", template=None, phases={}, problems=["no process template chosen (工程選択 is empty)"])
        self.assertEqual(1, run([row])["stats"]["deleted"])

    def test_a_pink_row_without_a_seiban_is_skipped(self):
        out = run([plan_row(kind=sl.DELETE, code=None, problems=["seiban 'x' is not a module code"])])
        self.assertEqual((0, 1), (out["stats"]["deleted"], out["stats"]["skipped"]))

    def test_the_same_seiban_on_two_pink_rows_is_deleted_once(self):
        out = run([plan_row(2, kind=sl.DELETE, code="530N00001A"), plan_row(3, kind=sl.DELETE, code="530N00001A")])
        self.assertEqual((1, 0), (out["stats"]["deleted"], out["stats"]["nothing_to_delete"]))
        self.assertTrue(any("handled once" in x for x in out["report"]))

    def test_a_seiban_on_a_white_and_a_pink_row_is_left_alone(self):
        out = run([plan_row(2, kind=sl.ADD, code="530N00001A"), plan_row(3, kind=sl.DELETE, code="530N00001A")])
        self.assertEqual(make_schedule(), out["schedule"])
        self.assertEqual(2, out["stats"]["skipped"])
        self.assertEqual(2, len(lines(out, "WARN")))

    def test_deleting_comes_before_adding(self):
        pink = plan_row(2, kind=sl.DELETE, code="530N00001A")
        white = plan_row(3, code="530N00005A")
        self.assertEqual(["530N00002A", "530N00005A"], codes(run([white, pink])))


class Other(unittest.TestCase):
    def test_grey_rows_are_ignored_and_counted(self):
        out = run([plan_row(kind=sl.KEEP, code="530N00001A"), plan_row(3, kind=sl.KEEP, code="530N09999A", customer="Cust Z")])
        self.assertEqual(2, out["stats"]["kept"])
        self.assertEqual((make_schedule(), {"environment": make_env()}), (out["schedule"], out["env"]))

    def test_the_warnings_of_the_workbook_come_first_in_the_report(self):
        out = run([plan_row()], warnings=["sheet '搬入先テーブル' not found: an empty Region cannot be filled from it"])
        self.assertEqual("WARN: seiban list: sheet '搬入先テーブル' not found: an empty Region cannot be filled from it", out["report"][0])
        self.assertEqual(1, out["stats"]["warnings"])

    def test_the_site_table_fills_an_empty_region(self):
        data = seiban_data([plan_row(customer="Cust C", fab="Fab C", region="")])
        data.sites.append(sl.Site("Site C", "Cust C", "Fab C", "Region Y"))
        data.__post_init__()
        out = app.apply_seiban_list(make_schedule(), {"environment": make_env()}, data)
        self.assertEqual("r2", out["env"]["environment"]["fab_list"][-1]["region"])

    def test_files_that_are_not_a_schedule_or_an_envconfig(self):
        with self.assertRaisesRegex(app.ApplyError, "no top-level 'schedule'"):
            app.apply_seiban_list({"environment": {}}, {"environment": make_env()}, seiban_data([]))
        with self.assertRaisesRegex(app.ApplyError, "no top-level 'environment'"):
            app.apply_seiban_list(make_schedule(), make_schedule(), seiban_data([]))

    def test_an_empty_list_changes_nothing(self):
        out = run([])
        self.assertEqual((make_schedule(), {"environment": make_env()}, []), (out["schedule"], out["env"], out["report"]))


# ----------------------------------------------------------------------------- the command line

PLAN_HEADER = ["登録日時", "製番", "装置名称", "顧客", "Fab", "Region", "開始可能日", "希望納期", "工程選択",
               "工程1_開始可能日", "工程1_終了期限日", "工程2_開始可能日", "工程2_終了期限日",
               "工程3_開始可能日", "工程3_終了期限日", "工程4_開始可能日", "工程4_終了期限日"]
FILLS = {"white": None, "grey": "FFD9D9D9", "pink": "FFFFDCE6"}


def sheet_row(code, template=1, customer="Cust A", fab="Fab A", region="Region X"):
    d = datetime
    return [d(2026, 11, 1), code, "Tool_X", customer, fab, region, None, None, f"工程テンプレート{template}",
            d(2026, 12, 1), d(2026, 12, 10), d(2026, 12, 5), d(2026, 12, 20), d(2026, 12, 15), d(2026, 12, 25), d(2026, 12, 22), d(2027, 1, 10)]


def write_workbook(path, rows, sheet="計画対象"):
    """rows: [(colour name, sheet row)]; templates 1 (= the operations of wf_tool) and 2 are written."""
    wb = Workbook()
    wb.active.title = "製番リスト"
    ws = wb.create_sheet(sheet)
    ws.append(PLAN_HEADER)
    for i, (colour, row) in enumerate(rows, start=2):
        ws.append(row)
        if FILLS[colour]:
            ws.cell(i, 2).fill = PatternFill("solid", fgColor=FILLS[colour])
    wb.create_sheet("搬入先テーブル").append(["搬入先", "顧客", "Fab", "Region"])
    for number, ops in ((1, TEMPLATE_LIKE_WF_TOOL), (2, TEMPLATE_WITH_P1)):
        t = wb.create_sheet(f"工程テンプレート{number}")
        t.append(["工程番号", "作業番号", "作業名", "工数", "最小人数", "最大人数"])
        for o in ops:
            t.append(list(o))
    wb.save(path)


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.schedule, self.env, self.book = self.dir / "01_Schedule.yaml", self.dir / "01_EnvConfig.yaml", self.dir / "list.xlsx"
        app.dump_yaml(self.schedule, make_schedule())
        app.dump_yaml(self.env, {"environment": make_env()})
        write_workbook(self.book, [("white", sheet_row("530N00005A", customer="Cust C", fab="Fab C")), ("pink", sheet_row("530N00001A")),
                                   ("grey", sheet_row("530N00002A"))])
        self.nothing = mock.patch.object(app, "CONFIG_FILE", self.dir / "no_config.txt")      # not the config.txt of the tool
        self.nothing.start()

    def tearDown(self):
        self.nothing.stop()
        self.td.cleanup()

    def main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = app.main([str(a) for a in argv])
        return rc, out.getvalue(), err.getvalue()

    def options(self, out_dir="out"):
        return ["--schedule", self.schedule, "--env", self.env, "--seiban-list", self.book, "--out-dir", self.dir / out_dir]

    def test_a_run_with_options_writes_the_two_files_and_leaves_the_inputs_alone(self):
        before = (self.schedule.read_bytes(), self.env.read_bytes())
        rc, out, err = self.main(*self.options())
        self.assertEqual((0, ""), (rc, err))
        self.assertEqual(["02_EnvConfig.yaml", "02_Schedule.yaml"], sorted(p.name for p in (self.dir / "out").iterdir()))
        sch = read_yaml(self.dir / "out" / "02_Schedule.yaml")["schedule"]
        self.assertEqual(["530N00002A", "530N00005A"], [t["name"] for t in sch["workflow_task_list"]])
        self.assertEqual(before, (self.schedule.read_bytes(), self.env.read_bytes()))
        self.assertIn("1 modules added, 1 deleted, 0 rows skipped, 1 grey rows ignored", out)

    def test_the_terminal_shows_counts_and_the_report_only_with_the_option(self):
        rc, out, _ = self.main(*self.options(), "--report")
        report = (self.dir / "out" / "02_apply_report.txt").read_text(encoding="utf-8")
        self.assertIn("module e3 added", report)
        self.assertIn("02_apply_report.txt", out)
        self.assertNotIn("530N00005A", out)                                           # no seiban in the terminal text
        rc, out, _ = self.main(*self.options("out2"))
        self.assertFalse((self.dir / "out2" / "02_apply_report.txt").exists())

    def test_the_warning_count_points_to_the_report(self):
        write_workbook(self.book, [("white", sheet_row("530N00002A"))])                # already in the Schedule
        rc, out, _ = self.main(*self.options())
        self.assertIn("1 warnings (run again with --report to write them to a file)", out)

    def config(self, **values):
        path = self.dir / "my_config.txt"
        path.write_text("\n".join(f"{k} = {v}" for k, v in values.items()) + "\n", encoding="utf-8")
        return path

    def test_everything_can_come_from_config_txt_and_paths_are_relative_to_it(self):
        cfg = self.config(schedule="01_Schedule.yaml", env="01_EnvConfig.yaml", seiban_list="list.xlsx", out_dir="from_config", report="true")
        rc, out, err = self.main("--config", cfg)
        self.assertEqual((0, ""), (rc, err))
        self.assertEqual(["02_EnvConfig.yaml", "02_Schedule.yaml", "02_apply_report.txt"], sorted(p.name for p in (self.dir / "from_config").iterdir()))

    def test_the_command_line_wins_over_config_txt(self):
        cfg = self.config(schedule="missing.yaml", env="missing.yaml", seiban_list="missing.xlsx", out_dir="from_config", report="true")
        rc, _, err = self.main("--config", cfg, *self.options("from_options"), "--no-report")
        self.assertEqual((0, ""), (rc, err))
        self.assertEqual(["02_EnvConfig.yaml", "02_Schedule.yaml"], sorted(p.name for p in (self.dir / "from_options").iterdir()))
        self.assertFalse((self.dir / "from_config").exists())

    def test_one_option_replaces_one_setting_of_config_txt(self):
        other = self.dir / "other.xlsx"
        write_workbook(other, [("white", sheet_row("530N00007A"))])
        cfg = self.config(schedule="01_Schedule.yaml", env="01_EnvConfig.yaml", seiban_list="list.xlsx", out_dir="out")
        self.main("--config", cfg, "--seiban-list", other)
        sch = read_yaml(self.dir / "out" / "02_Schedule.yaml")["schedule"]
        self.assertEqual("530N00007A", sch["workflow_task_list"][-1]["name"])

    def test_the_sheet_comes_from_the_option_or_config_txt(self):
        write_workbook(self.book, [("white", sheet_row("530N00005A"))], sheet="別シート")
        cfg = self.config(schedule="01_Schedule.yaml", env="01_EnvConfig.yaml", seiban_list="list.xlsx", out_dir="out", sheet="別シート")
        self.assertEqual(0, self.main("--config", cfg)[0])
        rc, _, err = self.main("--config", cfg, "--sheet", "計画対象")
        self.assertEqual(1, rc)
        self.assertIn("sheet '計画対象' not found", err)

    def test_the_output_folder_defaults_to_the_folder_of_the_schedule(self):
        sub = self.dir / "week"
        sub.mkdir()
        app.dump_yaml(sub / "01_Schedule.yaml", make_schedule())
        rc, _, _ = self.main("--schedule", sub / "01_Schedule.yaml", "--env", self.env, "--seiban-list", self.book)
        self.assertEqual(0, rc)
        self.assertTrue((sub / "02_Schedule.yaml").is_file())

    def test_no_input_at_all_says_where_to_set_it(self):
        rc, out, err = self.main()
        self.assertEqual(1, rc)
        self.assertIn("no schedule file: give --schedule or set schedule in", err)

    def test_a_missing_file_is_an_error_not_a_crash(self):
        rc, _, err = self.main(*self.options(), "--seiban-list", self.dir / "nope.xlsx")
        self.assertEqual(1, rc)
        self.assertIn("seiban-list file not found", err)

    def test_swapped_files_are_caught(self):
        rc, _, err = self.main("--schedule", self.env, "--env", self.schedule, "--seiban-list", self.book, "--out-dir", self.dir / "out")
        self.assertEqual(1, rc)
        self.assertIn("swapped", err)
        self.assertFalse((self.dir / "out").exists())

    def test_a_config_file_that_was_asked_for_must_exist(self):
        rc, _, err = self.main("--config", self.dir / "nope.txt")
        self.assertEqual(1, rc)
        self.assertIn("config file not found", err)

    def test_an_unknown_setting_is_noted(self):
        cfg = self.config(schedule="01_Schedule.yaml", env="01_EnvConfig.yaml", seiban_list="list.xlsx", out_dir="out", sheett="x")
        rc, _, err = self.main("--config", cfg)
        self.assertEqual(0, rc)
        self.assertIn("unknown setting 'sheett' is ignored", err)

    def test_a_bad_value_in_config_txt(self):
        cfg = self.config(schedule="01_Schedule.yaml", env="01_EnvConfig.yaml", seiban_list="list.xlsx", report="maybe")
        rc, _, err = self.main("--config", cfg)
        self.assertEqual(1, rc)
        self.assertIn("report must be true or false", err)

    def test_the_inputs_are_never_overwritten(self):
        app.dump_yaml(self.dir / "02_Schedule.yaml", make_schedule())
        rc, _, err = self.main("--schedule", self.dir / "02_Schedule.yaml", "--env", self.env, "--seiban-list", self.book, "--out-dir", self.dir)
        self.assertEqual(1, rc)
        self.assertIn("would be overwritten", err)

    def test_a_dry_run_writes_nothing(self):
        rc, out, _ = self.main(*self.options(), "--dry-run")
        self.assertEqual(0, rc)
        self.assertIn("Dry run: nothing written", out)
        self.assertFalse((self.dir / "out").exists())

    def test_a_file_that_did_not_change_is_copied_byte_for_byte(self):
        write_workbook(self.book, [("grey", sheet_row("530N00002A"))])
        self.main(*self.options())
        self.assertEqual(self.env.read_bytes(), (self.dir / "out" / "02_EnvConfig.yaml").read_bytes())
        self.assertEqual(self.schedule.read_bytes(), (self.dir / "out" / "02_Schedule.yaml").read_bytes())


class DeliveredConfig(unittest.TestCase):
    def test_the_config_txt_of_the_tool_uses_only_known_settings_and_has_no_paths_set(self):
        cfg = tfc.load(None, app.HERE / tfc.CONFIG_NAME, tfc.APPLY_SEIBAN_KEYS)
        self.assertEqual([], cfg.unknown)
        self.assertEqual(set(tfc.APPLY_SEIBAN_KEYS), set(cfg.values))                 # every setting is in the file
        self.assertIsNone(cfg.text("schedule"))
        self.assertTrue(cfg.bool("report"))


if __name__ == "__main__":
    unittest.main()
