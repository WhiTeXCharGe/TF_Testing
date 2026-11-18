"""The window of 01_StartSchedule: pick the two files, press Start, watch the log.  tkinter only (it comes with Python)."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import weeks
from .start import run_start

EXCEL_TYPES = [("Excel", "*.xlsm *.xlsx"), ("All files", "*.*")]


class StartWindow:
    def __init__(self, settings, day, su_others="", seiban_list="", submit=None):
        self.settings, self.day = settings, day
        self.queue = queue.Queue()
        self.worker = None
        self.root = tk.Tk()
        self.root.title("01 Start Schedule")
        self.root.minsize(760, 520)
        self.su_var = tk.StringVar(value=su_others or "")
        self.sb_var = tk.StringVar(value=seiban_list or "")
        self.submit_var = tk.BooleanVar(value=settings.flag("submit", True) if submit is None else submit)
        self._build()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------------------------------ layout
    def _build(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        name = weeks.week_name(self.day)
        previous = weeks.previous_week(self.settings.work_dir, name)
        info = [("Week", f"{name}  ({self.settings.work_dir / name})"),
                ("Previous week", previous.name if previous else "none: this is the first week (Master is used)")]
        if not previous:
            info.append(("Master", f"EnvConfig_start: {'found' if self.settings.master_start_env() else 'MISSING'}"
                                   f"   skill workbook: {'found' if self.settings.master_skill_file() else 'MISSING'}"))
        row = 0
        for label, text in info:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(frame, text=text).grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
            row += 1
        for label, var in (("SU_Others", self.su_var), ("Seiban list", self.sb_var)):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
            ttk.Button(frame, text="Browse...", command=lambda v=var, t=label: self._browse(v, t)).grid(row=row, column=2)
            row += 1
        ttk.Checkbutton(frame, text="Send to the solver at the end (2-6)", variable=self.submit_var).grid(
            row=row, column=1, sticky="w", pady=4)
        row += 1
        self.start_button = ttk.Button(frame, text="Start", command=self._start)
        self.start_button.grid(row=row, column=1, sticky="w", pady=8)
        self.status = ttk.Label(frame, text="")
        self.status.grid(row=row, column=1, sticky="e")
        row += 1
        frame.rowconfigure(row, weight=1)
        box = ttk.Frame(frame)
        box.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.log = tk.Text(box, height=16, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def _browse(self, var, title):
        path = filedialog.askopenfilename(title=f"Choose the {title} file", filetypes=EXCEL_TYPES)
        if path:
            var.set(path)

    # ------------------------------------------------------------------------------------------ run
    def _write(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        su, sb = self.su_var.get().strip(), self.sb_var.get().strip()
        if not su or not sb:
            messagebox.showwarning("01 Start Schedule", "Choose both files first: the SU_Others workbook and the seiban list.")
            return
        week = weeks.week_path(self.settings.work_dir, weeks.week_name(self.day))
        overwrite = False
        if week.exists() and weeks.was_submitted(week):
            if not messagebox.askyesno("01 Start Schedule", f"Week {week.name} was already sent to the solver.\n\n"
                                       "Run it again? The old files are kept (steps_old_*, logs_old_*)."):
                return
            overwrite = True
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.start_button.state(["disabled"])
        self.status.configure(text="running ...")
        submit = self.submit_var.get()

        def work():
            result = run_start(self.settings, su, sb, self.day, log=lambda line: self.queue.put(("line", line)),
                               submit=submit, overwrite=overwrite)
            self.queue.put(("done", result))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind, value = self.queue.get_nowait()
                if kind == "line":
                    self._write(value)
                else:
                    self._finished(value)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finished(self, result):
        self.start_button.state(["!disabled"])
        if result.ok:
            self.status.configure(text="finished")
            messagebox.showinfo("01 Start Schedule", result.message)
        else:
            self.status.configure(text="stopped")
            messagebox.showerror("01 Start Schedule", f"Stopped at {result.step}.\n\n{result.message}\n\nThe log is in the week folder (logs).")

    def run(self):
        self.root.mainloop()


def run_window(settings, day, su_others="", seiban_list="", submit=None):
    StartWindow(settings, day, su_others, seiban_list, submit).run()
