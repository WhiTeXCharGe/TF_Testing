"""plan_assist.config.txt: the settings of the programs the weekly user runs (01_StartSchedule, 02_UpdateSchedule).
Standard library only.

Same plain `key = value` format as the config.txt of the tools: blank lines and lines starting with # or ; are ignored, keys are not
case sensitive ("-" and "_" are the same), a value that is left empty counts as "not set", relative paths are relative to the
folder of the file.  A missing default file gives the defaults (folders next to the program).
"""
from __future__ import annotations

import re
from pathlib import Path

CONFIG_NAME = "plan_assist.config.txt"

# settings that are handed on to tool 2-6 (submit_to_timefold) when they are set
SOLVER_KEYS = ("mode", "azure_url", "local_url", "run_id", "params_file", "stage1_minutes", "stage2_enabled", "stage2_minutes",
               "stage1_unimproved_seconds", "stage1_stop_when_feasible", "stage2_unimproved_seconds", "random_seed",
               "allow_cut_fixed_trip")
KNOWN = ("tools_dir", "master_dir", "work_dir", "skill_file", "start_env", "su_sheet", "skill_sheet", "seiban_sheet",
         "default_skill_level", "submit", "tenant_id", "client_id") + SOLVER_KEYS
TRUE, FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


class SettingsError(Exception):
    """The settings file is missing (when asked for), unreadable or has a value that cannot be used."""


def parse_text(text):
    """-> (values, unknown keys)."""
    values, unknown = {}, []
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        eq = line.find("=")
        if eq <= 0:
            continue
        key = re.sub(r"[-\s]+", "_", line[:eq].strip().lower())
        value = line[eq + 1:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key not in KNOWN and key not in unknown:
            unknown.append(key)
        values[key] = value
    return values, unknown


class Settings:
    def __init__(self, values=None, path=None, base_dir=None, unknown=()):
        self.values = dict(values or {})
        self.path = Path(path) if path else None
        self.base_dir = Path(base_dir) if base_dir else (self.path.resolve().parent if self.path else Path.cwd())
        self.unknown = list(unknown)

    def has(self, key):
        return self.values.get(key, "") != ""

    def text(self, key, default=None):
        return self.values[key] if self.has(key) else default

    def folder(self, key, default=None):
        """A path of the settings, absolute against the folder of the file (default: a name relative to that folder)."""
        value = self.text(key, default)
        if value is None:
            return None
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.base_dir / p

    def flag(self, key, default):
        value = self.text(key)
        if value is None:
            return default
        if value.lower() in TRUE:
            return True
        if value.lower() in FALSE:
            return False
        raise SettingsError(f"{self._where()}: {key} must be true or false (got {value!r})")

    def whole(self, key):
        value = self.text(key)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            raise SettingsError(f"{self._where()}: {key} must be a whole number (got {value!r})") from None

    def _where(self):
        return str(self.path) if self.path else CONFIG_NAME

    @property
    def master_dir(self):
        return self.folder("master_dir", "Master")

    @property
    def work_dir(self):
        return self.folder("work_dir", "Work")

    @property
    def tools_dir(self):
        """The folder of the weekly_planning scripts: tools_dir, else nearby Weekly_planning, else a repository parent."""
        explicit = self.folder("tools_dir")
        if explicit:
            return explicit
        candidates = [
            self.base_dir / "Weekly_planning",
            self.base_dir.parent / "Weekly_planning",
            self.base_dir,
            self.base_dir.parent,
            self.base_dir.parent.parent,
        ]
        return next((c for c in candidates if (c / "common").is_dir()), candidates[0])

    def master_skill_file(self):
        """The skill workbook of the first week: skill_file, else the newest '*skill*' workbook in master_dir (None if none)."""
        explicit = self.folder("skill_file")
        if explicit:
            return explicit if explicit.is_file() else None
        found = sorted(p for p in self.master_dir.glob("*skill*.xls*") if not p.name.startswith("~$")) if self.master_dir.is_dir() else []
        return found[-1] if found else None

    def master_start_env(self):
        """Master/EnvConfig_start.yaml of the first week (None if it is not there)."""
        p = Path(self.text("start_env", "EnvConfig_start.yaml")).expanduser()
        p = p if p.is_absolute() else self.master_dir / p          # a plain name lives in master_dir
        return p if p.is_file() else None


def load(path=None, program_dir=None):
    """The Settings of `path`, else of plan_assist.config.txt in `program_dir`, the folder of the program (a missing default file
    gives the defaults, with the folders next to the program)."""
    explicit = path is not None
    program_dir = Path(program_dir) if program_dir else Path.cwd()
    p = Path(path) if explicit else program_dir / CONFIG_NAME
    if not p.is_file():
        if explicit:
            raise SettingsError(f"settings file not found: {p}")
        return Settings(base_dir=program_dir)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SettingsError(f"cannot read {p}: {e}") from None
    values, unknown = parse_text(text)
    return Settings(values, p, unknown=unknown)