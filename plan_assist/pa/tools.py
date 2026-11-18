"""Runs the tools of weekly_planning from their files, in this process, as main(argv).

The tool folders have names such as 2-2-3prepare_schedule that cannot be imported by name, so each script is loaded by its path.
The output of a tool goes line by line to `emit` while it runs; an exception or an exit of the tool never escapes.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# step -> (folder, script) inside the weekly_planning folder
TOOLS = {
    "2-2-3": ("2-2-3prepare_schedule", "prepare_schedule.py"),
    "2-4": ("2-4apply_seiban_list", "apply_seiban_list.py"),
    "2-5": ("2-5set_plan_period", "set_plan_period.py"),
    "2-6": ("2-6submit_to_timefold", "submit_to_timefold.py"),
    "3-1": ("3-1receive_from_timefold", "receive_from_timefold.py"),
    "3-2": ("3-2extract_flexible_schedule", "extract_flexible_schedule.py"),
    "3-3": ("3-3write_su_others", "write_su_others.py"),
}


@dataclass
class ToolResult:
    rc: int
    lines: list = field(default_factory=list)          # everything the tool printed
    errors: list = field(default_factory=list)         # what it printed as an error

    def last_error(self):
        """The most useful line to tell the user why the tool stopped."""
        if self.errors:
            return self.errors[-1]
        for line in reversed(self.lines):
            if line.strip():
                return line.strip()
        return f"exit code {self.rc}"


class _Sink(io.TextIOBase):
    """A text stream that hands every finished line to a callback."""

    def __init__(self, callback):
        self.callback = callback
        self._buffer = ""

    def writable(self):
        return True

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.callback(line.rstrip("\r"))
        return len(text)

    def flush(self):
        pass

    def finish(self):
        if self._buffer.strip():
            self.callback(self._buffer.rstrip("\r\n"))
        self._buffer = ""


class ToolRunner:
    def __init__(self, tools_dir):
        self.tools_dir = Path(tools_dir)
        self._modules = {}

    def script(self, step):
        folder, name = TOOLS[step]
        return self.tools_dir / folder / name

    def missing(self, steps):
        """What is not found in the weekly_planning folder for these steps (empty list: all there)."""
        problems = []
        if not (self.tools_dir / "common").is_dir():
            problems.append(f"{self.tools_dir / 'common'} (the shared code)")
        problems += [str(self.script(s)) for s in steps if not self.script(s).is_file()]
        return problems

    def _load(self, step):
        if step not in self._modules:
            path = self.script(step)
            spec = importlib.util.spec_from_file_location(f"pa_tool_{step.replace('-', '_')}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._modules[step] = module
        return self._modules[step]

    def run(self, step, argv, emit=None):
        """Calls main(argv) of the tool. Returns ToolResult (rc 1 when the tool raised an exception or did not load)."""
        emit = emit or (lambda line: None)
        lines, errors = [], []

        def out(line):
            lines.append(line)
            emit(line)

        def err(line):
            lines.append(line)
            errors.append(line)
            emit(line)

        stdout, stderr = _Sink(out), _Sink(err)
        rc = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = self._load(step).main([str(a) for a in argv])
            rc = 0 if result is None else int(result)
        except SystemExit as e:                                  # argparse errors, sys.exit in a tool
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception as e:                                   # a tool must never take the program down
            rc = 1
            errors.append(f"{type(e).__name__}: {e}")
            for line in traceback.format_exc().rstrip().splitlines():
                lines.append(line)
                emit(line)
        finally:
            stdout.finish()
            stderr.finish()
        return ToolResult(rc, lines, errors)
