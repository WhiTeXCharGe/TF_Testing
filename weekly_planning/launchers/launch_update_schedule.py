#!/usr/bin/env python3
"""
launch_update_schedule.py  --  the entry point of 02_UpdateSchedule.exe (and a way to start the window from a console).

The program itself is update_schedule/gui.py. This file makes the scripts folder importable and shows a start-up error to a user
who has no console.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# A windowed exe has no console: sys.stdout / sys.stderr are None. Remember that, and give the program something to write to.
NO_CONSOLE = sys.stdout is None or sys.stderr is None
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

TITLE = "02_UpdateSchedule 計画結果の反映"


def app_dir():
    """The folder of the exe, or weekly_planning/ when started from the repository."""
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


APP = app_dir()
sys.path.insert(0, str(APP / "weekly_planning" if (APP / "weekly_planning" / "common").is_dir() else APP))


def fail_visibly(text):
    """An error message the user can see: on stderr, and (windowed exe: nobody sees stderr) in plan_assist_error.txt and a message box."""
    print(text, file=sys.stderr)
    if not NO_CONSOLE:
        return
    try:
        (APP / "plan_assist_error.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(TITLE, text)
        root.destroy()
    except Exception:
        pass


def main():
    try:
        from update_schedule.gui import main as run_window
        run_window()
    except Exception:
        fail_visibly("予期しないエラーのため、プログラムが停止しました。\n"
                     "weekly_planning フォルダに common と update_schedule があるか確認してください。\n\n" + traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())