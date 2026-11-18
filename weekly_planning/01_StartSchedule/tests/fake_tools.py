"""A small fake weekly_planning folder for the tests: every tool records how it was called and writes the files the real one writes.

A tool fails when a file `fail_<step>` (text = what it prints as the error) exists in the folder; `raise_<step>` makes it raise.
"""
from __future__ import annotations

import json
from pathlib import Path

HELPERS = '''
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(step, argv):
    log = ROOT / "calls.json"
    calls = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    calls.append({"step": step, "argv": [str(a) for a in argv]})
    log.write_text(json.dumps(calls), encoding="utf-8")


def opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def cfg(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def trouble(step):
    import sys
    if (ROOT / f"raise_{step}").exists():
        raise RuntimeError("tool exploded")
    f = ROOT / f"fail_{step}"
    if f.exists():
        print("some progress")
        print(f.read_text(encoding="utf-8"), file=sys.stderr)
        return 1
    return None
'''

SCRIPTS = {
    "2-2-3prepare_schedule/prepare_schedule.py": '''

def main(argv):
    record("2-2-3", argv)
    print("reading SU_Others (this takes a while) ...")
    bad = trouble("2-2-3")
    if bad:
        return bad
    out = Path(opt(argv, "--out-dir"))
    out.mkdir(parents=True, exist_ok=True)
    day = opt(argv, "--today").replace("/", "")
    for name in (day + "_FlexibleWork.yaml", "01_EnvConfig.yaml", "01_Schedule.yaml"):
        (out / name).write_text("x\\n", encoding="utf-8")
    return 0
''',
    "2-4apply_seiban_list/apply_seiban_list.py": '''

def main(argv):
    record("2-4", argv)
    bad = trouble("2-4")
    if bad:
        return bad
    out = Path(cfg(opt(argv, "--config"))["out_dir"])
    for name in ("02_Schedule.yaml", "02_EnvConfig.yaml"):
        (out / name).write_text("x\\n", encoding="utf-8")
    return 0
''',
    "2-5set_plan_period/set_plan_period.py": '''

def main(argv):
    record("2-5", argv)
    bad = trouble("2-5")
    if bad:
        return bad
    for name in ("--output-schedule", "--output-env-config"):
        Path(opt(argv, name)).write_text("x\\n", encoding="utf-8")
    return 0
''',
    "2-6submit_to_timefold/submit_to_timefold.py": '''

def main(argv):
    record("2-6", argv)
    bad = trouble("2-6")
    if bad:
        return bad
    out = Path(cfg(opt(argv, "--config"))["out_dir"])
    (out / "run_info.json").write_text(json.dumps({"runId": "run-test"}), encoding="utf-8")
    print("Submitted: run run-test")
    return 0
''',
}


def make_tools(folder):
    """Creates the fake weekly_planning folder in `folder` and returns it."""
    folder = Path(folder)
    (folder / "common").mkdir(parents=True, exist_ok=True)
    (folder / "common" / "__init__.py").write_text("", encoding="utf-8")
    for rel, text in SCRIPTS.items():           # the helpers are copied into every script: a shared module would be cached
        p = folder / rel                        # between tests that use different folders
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(HELPERS + text, encoding="utf-8")
    return folder


def calls(folder):
    p = Path(folder) / "calls.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
