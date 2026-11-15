#!/usr/bin/env python3
"""
apply_seiban_list.py  --  weekly planning tool "2-4": add the new seiban of the week to the plan.

Run it with no options: everything is read from the config.txt next to this script (input files, seiban list, output folder).
Any command line option replaces the same setting of config.txt for that run.

Input   01_Schedule.yaml, 01_EnvConfig.yaml   (from tool 2-2-3)   and   the seiban list workbook (sheet 計画対象)
Output  02_Schedule.yaml, 02_EnvConfig.yaml   (and 02_apply_report.txt with --report)

Row colour of the seiban cell:  white = new seiban (add a module),  grey = already planned (ignored),  pink = delete its module.
  EnvConfig  customers, fabs, regions, workflows (process templates) and worker skills that the new seiban need
  Schedule   one workflow task per white row; the workflow task and the assignment rows of a pink row are removed
Nothing is ever deleted from the EnvConfig.  The terminal shows counts; every warning is in the report (--report).
The input files are never changed.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))      # weekly_planning/ (the folder of common)
sys.path.insert(0, str(HERE))             # the modules of this tool
from common import seiban_list as sl, tf_config as tfc   # noqa: E402
from common.yaml_io import dump_yaml, load_yaml   # noqa: E402
import env_masters as masters   # noqa: E402
import env_skills as skills   # noqa: E402
import env_workflows as workflows   # noqa: E402
import schedule_modules as modules   # noqa: E402

CONFIG_FILE = HERE / tfc.CONFIG_NAME        # the settings of this tool
OUT_SCHEDULE, OUT_ENV, OUT_REPORT = "02_Schedule.yaml", "02_EnvConfig.yaml", "02_apply_report.txt"


class ApplyError(ValueError):
    """An input file is missing or is not the kind of file it should be."""


def _label(row):
    return f"row {row.excel_row} ({row.code or row.raw_code or 'no seiban'})"


def apply_seiban_list(schedule_doc, env_doc, data):
    """schedule_doc / env_doc: parsed 01_Schedule.yaml / 01_EnvConfig.yaml.  data: a SeibanList.
    Returns {"schedule", "env", "report" [lines], "stats"}.  The inputs are not modified."""
    if not isinstance(schedule_doc, dict) or not isinstance(schedule_doc.get("schedule"), dict):
        raise ApplyError("the schedule file has no top-level 'schedule'. Are schedule and env swapped?")
    if not isinstance(env_doc, dict) or not isinstance(env_doc.get("environment"), dict):
        raise ApplyError("the EnvConfig file has no top-level 'environment'. Are schedule and env swapped?")
    out_schedule, out_env = copy.deepcopy(schedule_doc), copy.deepcopy(env_doc)
    sch, env = out_schedule["schedule"], out_env["environment"]
    before = {k: len(env.get(k) or []) for k in ("customer_company_list", "fab_list", "region_list", "workflow_list")}
    report = [f"WARN: seiban list: {w}" for w in data.warnings]
    stats = {"added": 0, "deleted": 0, "skipped": 0, "kept": len(data.rows_of(sl.KEEP)), "nothing_to_delete": 0}

    # ---- 1. sort the rows
    white, pink = [], []
    for row in data.rows:
        if row.kind == sl.UNKNOWN:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: {'; '.join(row.problems)}")
        elif row.kind == sl.ADD:
            white.append(row)
        elif row.kind == sl.DELETE:
            pink.append(row)
    both = {r.code for r in white if r.code} & {r.code for r in pink if r.code}
    for row in white + pink:
        if row.code in both:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: the same seiban is also on a {'pink' if row.kind == sl.ADD else 'white'} row")
    white = [r for r in white if r.code not in both]
    pink = [r for r in pink if r.code not in both]

    # ---- 2. delete (pink)
    done = set()
    for row in pink:
        problems = modules.delete_row_problems(row)
        if problems:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: {'; '.join(problems)}")
            continue
        if row.code in done:
            report.append(f"INFO: {_label(row)}: the same seiban is on more than one pink row; it was handled once")
            continue
        done.add(row.code)
        tasks = modules.module_tasks(sch, row.code)
        if not tasks:
            stats["nothing_to_delete"] += 1
            report.append(f"INFO: {_label(row)}: no workflow task in the Schedule (misc task, Fixed, past or never planned): nothing to delete")
            continue
        for task in tasks:
            gone = modules.delete_module(sch, task)
            stats["deleted"] += 1
            report.append(f"INFO: {_label(row)}: module {task['id']} deleted ({gone['phases']} phase tasks, {gone['operations']} operation tasks, "
                          f"{gone['rows']} assignment rows)")
            if gone["fixed_rows"]:
                report.append(f"WARN: {_label(row)}: {gone['fixed_rows']} of the deleted assignment rows were Fixed")

    # ---- 3. add (white): masters, workflow and skills first, then the module
    refs, seen = {}, set()
    for row in white:
        if row.code in seen:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: the same seiban is on an earlier white row")
            continue
        if row.code:
            seen.add(row.code)
        existing = modules.module_tasks(sch, row.code) if row.code else []
        if existing:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: already a workflow task in the Schedule ({existing[0]['id']})")
            continue
        template = data.templates.get(row.template)
        problems = modules.add_row_problems(row, template)
        if problems:
            stats["skipped"] += 1
            report.append(f"WARN: {_label(row)}: skipped: {'; '.join(problems)}")
            continue
        fab_id = masters.resolve_site(env, row.customer, row.fab, row.region, data.site_region(row.customer, row.fab), report)
        if row.template not in refs:
            refs[row.template] = workflows.ensure_workflow(env, template, report)
            if refs[row.template].created:
                skills.add_skills(env, refs[row.template].new_ops, report)
        task = modules.add_module(sch, row.code, refs[row.template], fab_id, template, row)
        stats["added"] += 1
        report.append(f"INFO: {_label(row)}: module {task['id']} added (workflow {task['workflow']}, fab {fab_id})")

    after = {k: len(env.get(k) or []) for k in before}
    stats.update(customers=after["customer_company_list"] - before["customer_company_list"], fabs=after["fab_list"] - before["fab_list"],
                 regions=after["region_list"] - before["region_list"], workflows=after["workflow_list"] - before["workflow_list"],
                 warnings=sum(1 for line in report if line.startswith("WARN")))
    return {"schedule": out_schedule, "env": out_env, "report": report, "stats": stats}


def summary_lines(stats):
    return [f"seiban list: {stats['added']} modules added, {stats['deleted']} deleted, {stats['skipped']} rows skipped, "
            f"{stats['kept']} grey rows ignored, {stats['nothing_to_delete']} pink rows without a module",
            f"EnvConfig: {stats['customers']} customers, {stats['fabs']} fabs, {stats['regions']} regions, {stats['workflows']} workflows added"]


def first(*values):
    """The first value that is not None (command line, then config.txt, then the default)."""
    for v in values:
        if v is not None:
            return v
    return None


def _write(doc, original, changed, path):
    """The new file: a copy of the input when nothing changed in it, else the dumped yaml."""
    if changed:
        dump_yaml(path, doc)
    else:
        shutil.copyfile(original, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Weekly planning 2-4: add the new seiban of the seiban list to the plan. "
                                             "Settings come from config.txt; the options below replace them for one run.")
    ap.add_argument("--config", default=None, help=f"settings file (default: {tfc.CONFIG_NAME} next to this script)")
    ap.add_argument("--schedule", default=None, help="01_Schedule.yaml (config: schedule)")
    ap.add_argument("--env", default=None, help="01_EnvConfig.yaml (config: env)")
    ap.add_argument("--seiban-list", default=None, help="the seiban list workbook (config: seiban_list)")
    ap.add_argument("--out-dir", default=None, help=f"folder for {OUT_SCHEDULE} and {OUT_ENV} (config: out_dir; default: the folder of the schedule)")
    ap.add_argument("--sheet", default=None, help=f"sheet with the rows to handle (config: sheet; default {sl.PLAN_SHEET})")
    ap.add_argument("--report", action=argparse.BooleanOptionalAction, default=None,
                    help=f"also write {OUT_REPORT}: every change and every warning (config: report)")
    ap.add_argument("--dry-run", action="store_true", help="read and check everything, write nothing")
    args = ap.parse_args(argv)

    try:
        cfg = tfc.load(args.config, CONFIG_FILE, tfc.APPLY_SEIBAN_KEYS)
        for key in cfg.unknown:
            print(f"Note: {cfg.path.name}: unknown setting '{key}' is ignored.", file=sys.stderr)
        schedule = first(args.schedule, cfg.file("schedule"))
        env = first(args.env, cfg.file("env"))
        workbook = first(args.seiban_list, cfg.file("seiban_list"))
        for label, value, key in (("schedule", schedule, "schedule"), ("env", env, "env"), ("seiban-list", workbook, "seiban_list")):
            if not value:
                raise ApplyError(f"no {label} file: give --{label} or set {key} in {cfg.path or tfc.CONFIG_NAME}")
            if not Path(value).is_file():
                raise ApplyError(f"{label} file not found: {value}")
        sheet = first(args.sheet, cfg.text("sheet"), sl.PLAN_SHEET)
        write_report = first(args.report, cfg.bool("report"), False)
        out_arg = first(args.out_dir, cfg.file("out_dir"))
        out_dir = Path(out_arg) if out_arg else Path(schedule).resolve().parent
        for name, source in ((OUT_SCHEDULE, schedule), (OUT_ENV, env)):
            if (out_dir / name).resolve() == Path(source).resolve():
                raise ApplyError(f"{out_dir / name} is also an input file and would be overwritten: use another --out-dir")

        print(f"Settings: {cfg.path or 'no config file, command line only'}")
        print(f"Reading {Path(workbook).name} (sheet {sheet}) ...")
        data = sl.read_seiban_list(workbook, plan_sheet=sheet)
        schedule_doc, env_doc = load_yaml(schedule), load_yaml(env)
        result = apply_seiban_list(schedule_doc, env_doc, data)
    except (tfc.ConfigError, sl.SeibanListError, ApplyError, OSError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    stats, report = result["stats"], result["report"]
    lines = summary_lines(stats)
    print("\n".join(lines))
    if stats["warnings"]:
        print(f"{stats['warnings']} warnings" + ("" if write_report else " (run again with --report to write them to a file)"))
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(result["schedule"], schedule, result["schedule"] != schedule_doc, out_dir / OUT_SCHEDULE)
    _write(result["env"], env, result["env"] != env_doc, out_dir / OUT_ENV)
    names = [OUT_SCHEDULE, OUT_ENV]
    if write_report:
        head = ["apply_seiban_list report", f"schedule: {Path(schedule).name}, EnvConfig: {Path(env).name}, "
                f"seiban list: {Path(workbook).name} (sheet {sheet})"]
        (out_dir / OUT_REPORT).write_text("\n".join(head + lines + [""] + report) + "\n", encoding="utf-8")
        names.append(OUT_REPORT)
    print(f"written to {out_dir}: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
