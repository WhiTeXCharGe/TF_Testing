#!/usr/bin/env python3
"""
build_exe.py  --  builds 01_StartSchedule.exe (PyInstaller, one folder) and assembles the plan_assist folder.

    python build_exe.py [--out <folder>] [--console]

Needs PyInstaller (python -m pip install pyinstaller). Result: <out>/plan_assist/ with

    01_StartSchedule.exe + _internal/   the program (Python and the libraries are inside)
    config\plan_assist.config.txt       the settings (sample; edit it)
    Weekly_planning/                    a copy of the tools (common/ and the tool folders, without tests and without their config.txt)
    Master/                             where the first-week files go
    Work/                               one folder per week is made here

The tools are NOT inside the exe: the program loads them from Weekly_planning/ at run time, so a tool can be updated by replacing its
folder, without a new build. What is inside the exe: Python, tkinter and the libraries the tools import (found by reading every
tool; a tool that starts to import a new library needs a new build).  `common` is left out of the exe on purpose: the exe and the tools
must use the same `common` from Weekly_planning/.

--console builds a version with a console window (it shows the log and errors), for finding problems.
"""
from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent                                     # weekly_planning/
NAME = "01_StartSchedule"
SKIP_DIRS = {"tests", "__pycache__", "dist", "build", NAME, "config"}  # not part of the tools copy
SKIP_FILES = {"config.txt"}                                          # the settings of a developer's PC: never shipped
BIG_AND_UNUSED = ("matplotlib", "scipy", "IPython", "pytest", "numba", "PIL", "tornado", "jinja2", "sqlalchemy", "lxml")


def python_files(root):
    """Every .py file of the tools (not tests, not this program)."""
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel[:-1]) or p.name == "build_exe.py":
            continue
        yield p


def imported_modules(root):
    """Top-level names imported anywhere in the tools that are not the tools' own modules: the libraries the exe must contain."""
    files = list(python_files(root))
    own = {p.stem for p in files} | {p.parent.name for p in files if p.name == "__init__.py"} | {"common"}
    found = set()
    for p in files:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return sorted(found - own - {"__future__"})


def copy_tools(dest):
    """A copy of weekly_planning for the user's PC."""
    def ignore(folder, names):
        skipped = [n for n in names if n in SKIP_DIRS or n in SKIP_FILES or n.endswith(".pyc")]
        if Path(folder).resolve() == TOOLS:
            skipped += [n for n in names if (TOOLS / n).is_dir() and not any((TOOLS / n).glob("*.py")) and n != "common" and n not in skipped]
        return skipped
    shutil.copytree(TOOLS, dest, ignore=ignore)


def build(out, console):
    try:
        import PyInstaller   # noqa: F401
    except ImportError:
        sys.exit("PyInstaller is not installed: python -m pip install pyinstaller")
    libraries = imported_modules(TOOLS)
    print("libraries the tools import:", ", ".join(libraries))
    out = Path(out).resolve()
    work = Path(tempfile.mkdtemp(prefix="plan_assist_build_"))
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", NAME, "--onedir",
           "--console" if console else "--windowed",
           "--distpath", str(work / "dist"), "--workpath", str(work / "work"), "--specpath", str(work),
           "--paths", str(HERE), "--exclude-module", "common",
           "--collect-submodules", "openpyxl", "--collect-submodules", "yaml"]
    for lib in libraries:
        cmd += ["--hidden-import", lib]
    for lib in BIG_AND_UNUSED:
        cmd += ["--exclude-module", lib]
    cmd.append(str(HERE / "start_schedule.py"))
    print("running PyInstaller ...", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"PyInstaller failed (exit code {result.returncode}); the build folder is {work}")

    target = out / "plan_assist"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(work / "dist" / NAME, target)
    copy_tools(target / "Weekly_planning")
    config_text = (HERE.parent / "config" / "plan_assist.config.txt").read_text(
        encoding="utf-8"
    )
    config_text = config_text.replace(
        "master_dir = ..\\..\\Master", "master_dir = ..\\Master"
    ).replace("work_dir = ..\\..\\Work", "work_dir = ..\\Work")
    config_directory = target / "config"
    config_directory.mkdir()
    (config_directory / "plan_assist.config.txt").write_text(
        config_text, encoding="utf-8"
    )
    (target / "Master").mkdir()
    (target / "Master" / "README.txt").write_text(
        "First week only: put EnvConfig_start.yaml and the skill workbook (a file with 'skill' in its name) here.\n"
        "最初の週だけ: EnvConfig_start.yaml とスキルレベルのファイル(名前に skill を含むもの)をここに置く。\n", encoding="utf-8")
    (target / "Work").mkdir()
    shutil.rmtree(work, ignore_errors=True)
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
    print(f"\nBuilt: {target / (NAME + '.exe')}   (the whole folder: {size:.0f} MB)")
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build 01_StartSchedule.exe with PyInstaller and assemble the plan_assist folder.")
    ap.add_argument("--out", default=str(HERE / "dist"), help="folder for the result (default: dist next to this script)")
    ap.add_argument("--console", action="store_true", help="build with a console window (for finding problems)")
    args = ap.parse_args(argv)
    build(args.out, args.console)
    return 0


if __name__ == "__main__":
    sys.exit(main())