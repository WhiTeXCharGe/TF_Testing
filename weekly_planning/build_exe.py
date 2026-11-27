#!/usr/bin/env python3
"""
build_exe.py  --  builds 01_StartSchedule.exe and 02_UpdateSchedule.exe (PyInstaller, one folder) and assembles the plan_assist folder.

    python build_exe.py [--out <folder>] [--console] [--only 01|02] [--allow-missing]

Needs PyInstaller (python -m pip install pyinstaller) and, because SharePoint links are used, msal and requests
(python -m pip install msal requests). Result: <out>/plan_assist/ with

    01_StartSchedule.exe                        part 1
    02_UpdateSchedule.exe                       part 2
    config/plan_assist.config.txt               the settings of both programs (sample; edit it)
    weekly_planning/                            Python, libraries, and the scripts the programs run
    Master/                                     where the first-week files go
    Work/                                       one folder per week is made here

The scripts are NOT inside the exes: the programs load them from weekly_planning at run time, so a script can be updated
by replacing its folder, without a new build. What is inside an exe: Python, tkinter and the libraries the scripts import (found by
reading every script; a script that starts to import a new library needs a new build). `common` and `update_schedule` are left out
of the exes on purpose: both programs and all tools must use the same code from weekly_planning.

--console builds versions with a console window (they show the log and errors), for finding problems.
--only builds one of the two programs (the other one is left out of the result folder).
--allow-missing builds even when a library the scripts import is not installed here (that exe will not work with SharePoint links).
"""
from __future__ import annotations

import argparse
import ast
import filecmp
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # weekly_planning/
PROGRAMS = {
    "01": {"name": "01_StartSchedule", "entry": HERE / "01_StartSchedule" / "start_schedule.py",
           "paths": [HERE / "01_StartSchedule"]},
    "02": {"name": "02_UpdateSchedule", "entry": HERE / "launchers" / "launch_update_schedule.py",
           "paths": []},
}
CONTENTS = "weekly_planning"
NOT_TOOLS = {"tests", "testing", "__pycache__", "dist", "build", "config", "doc", "launchers", "01_StartSchedule"}    # not copied, not scanned
SKIP_FILES = {"config.txt"}                                          # the settings of a developer's PC: never shipped
BIG_AND_UNUSED = ("matplotlib", "scipy", "IPython", "pytest", "numba", "PIL", "tornado", "jinja2", "sqlalchemy", "lxml")
LEFT_OUT = ("common", "update_schedule")                             # read from disk next to the exe, never from inside an exe
MASTER_README = ("First week only: put EnvConfig_start.yaml and the skill workbook (a file with 'skill' in its name) here.\n"
                 "最初の週だけ: EnvConfig_start.yaml とスキルレベルのファイル(名前に skill を含むもの)をここに置く。\n")


def tool_folders():
    """The folders of the scripts that are copied next to the exes: common, the tools, update_schedule."""
    return sorted(d for d in HERE.iterdir()
                  if d.is_dir() and d.name not in NOT_TOOLS and not d.name.startswith(".") and (d.name == "common" or any(d.glob("*.py"))))


def python_files():
    """Every .py file of the scripts (not tests, not the programs' own entry files)."""
    for folder in tool_folders():
        for p in sorted(folder.rglob("*.py")):
            if not any(part in NOT_TOOLS for part in p.relative_to(HERE).parts[:-1]):
                yield p


def _exists(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def imported_modules():
    """The modules imported anywhere in the scripts that are not the scripts' own (dotted names, e.g. tkinter.filedialog): the scripts
    are loaded from disk at run time, so PyInstaller cannot see these imports and must be told."""
    files = list(python_files())
    own = {p.stem for p in files} | {d.name for d in tool_folders()} | set(LEFT_OUT) | {"__future__"}
    found = set()
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module)
                found.update(f"{node.module}.{a.name}" for a in node.names if _exists(f"{node.module}.{a.name}"))
    return sorted(name for name in found if name.split(".")[0] not in own)


def not_installed(libraries):
    """The top-level libraries that are neither part of Python nor installed here."""
    tops = sorted({name.split(".")[0] for name in libraries})
    return [name for name in tops if name not in sys.stdlib_module_names and importlib.util.find_spec(name) is None]


def copy_tools(dest):
    """The scripts in the shared runtime folder."""
    def ignore(folder, names):
        return [n for n in names if n in NOT_TOOLS or n in SKIP_FILES or n.endswith(".pyc")]
    for folder in tool_folders():
        shutil.copytree(folder, dest / folder.name, ignore=ignore)


def merge_runtime(source, dest):
    """Combine the two runtimes only when their shared files are compatible."""
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = dest / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        elif not filecmp.cmp(path, target, shallow=False):
            if relative == Path("base_library.zip"):
                with zipfile.ZipFile(path) as a, zipfile.ZipFile(target) as b:
                    names = set(a.namelist())
                    if names == set(b.namelist()) and all(a.read(name) == b.read(name) for name in names):
                        continue
            raise RuntimeError(f"Conflicting runtime files: {relative}. Cannot safely combine the executables.")


def pyinstaller(program, libraries, work, console):
    """One PyInstaller run; returns the folder it made (<work>/dist/<name>)."""
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", program["name"], "--onedir",
           "--console" if console else "--windowed", "--contents-directory", CONTENTS,
           "--distpath", str(work / "dist"), "--workpath", str(work / "work" / program["name"]), "--specpath", str(work)]
    for path in program["paths"]:
        cmd += ["--paths", str(path)]
    for module in LEFT_OUT:
        cmd += ["--exclude-module", module]
    cmd += ["--collect-submodules", "openpyxl", "--collect-submodules", "yaml"]
    for lib in libraries:
        cmd += ["--hidden-import", lib]
    for lib in BIG_AND_UNUSED:
        cmd += ["--exclude-module", lib]
    cmd.append(str(program["entry"]))
    print(f"\n=== {program['name']}: running PyInstaller ...", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"PyInstaller failed for {program['name']} (exit code {result.returncode}); the build folder is {work}")
    return work / "dist" / program["name"]


def build(out, console, only, allow_missing):
    try:
        import PyInstaller   # noqa: F401
    except ImportError:
        sys.exit("PyInstaller is not installed: python -m pip install pyinstaller")
    libraries = imported_modules()
    print("libraries the scripts import:", ", ".join(libraries))
    missing = not_installed(libraries)
    if missing:
        text = (f"not installed here: {', '.join(missing)}  (python -m pip install {' '.join(missing)})")
        if not allow_missing:
            sys.exit(f"Cannot build a working exe: {text}\nInstall them, or use --allow-missing to build anyway.")
        print(f"WARNING: {text}\n         the exes are built without them; SharePoint links will not work in them.")
    selected = [PROGRAMS[key] for key in (("01", "02") if only is None else (only,))]

    out = Path(out).resolve()
    target = out / "plan_assist"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="plan_assist_build_"))
    for program in selected:
        made = pyinstaller(program, libraries, work, console)
        shutil.copy2(made / f"{program['name']}.exe", target)
        merge_runtime(made / CONTENTS, target / CONTENTS)

    copy_tools(target / CONTENTS)
    config_text = (HERE / "config" / "plan_assist.config.txt").read_text(encoding="utf-8")
    config_text = config_text.replace("master_dir = ..\\..\\Master", "master_dir = ..\\Master").replace(
        "work_dir = ..\\..\\Work", "work_dir = ..\\Work")
    (target / "config").mkdir()
    (target / "config" / "plan_assist.config.txt").write_text(config_text, encoding="utf-8")
    (target / "Master").mkdir()
    (target / "Master" / "README.txt").write_text(MASTER_README, encoding="utf-8")
    (target / "Work").mkdir()
    shutil.rmtree(work, ignore_errors=True)

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    print(f"\nBuilt in {target}   (the whole folder: {size:.0f} MB)")
    for program in selected:
        print(f"  {target / (program['name'] + '.exe')}")
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build 01_StartSchedule.exe and 02_UpdateSchedule.exe with PyInstaller and assemble the "
                                             "plan_assist folder.")
    ap.add_argument("--out", default=str(HERE / "dist"), help="folder for the result (default: dist next to this script)")
    ap.add_argument("--console", action="store_true", help="build with console windows (for finding problems)")
    ap.add_argument("--only", choices=("01", "02"), default=None, help="build only this program")
    ap.add_argument("--allow-missing", action="store_true", help="build even if a library the scripts import is not installed")
    args = ap.parse_args(argv)
    build(args.out, args.console, args.only, args.allow_missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())