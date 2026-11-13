"""
tf_config.py -- reads config.txt, the settings file of tools 2-6 (submit) and 3-1 (receive).  Standard library only.
Each tool has its own config.txt in its own folder; this module is the shared reader.

Same plain `key=value` format as the config.txt of the SchedulerWeb app: blank lines and lines starting with # or ; are ignored,
keys are not case sensitive ("-" and "_" are the same), values may be quoted, and a value that is left empty counts as "not set".
Relative paths are relative to the folder that holds config.txt.

Order of priority everywhere: command line option > config.txt > built-in default.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from . import timefold_api as api

CONFIG_NAME = "config.txt"

# The settings each tool knows. Any other key is reported (a typo would otherwise be silently ignored).
CONNECTION_KEYS = ("mode", "azure_url", "local_url")
SUBMIT_KEYS = CONNECTION_KEYS + (
    "schedule", "env", "out_dir", "run_id", "params_file",
    "stage1_minutes", "stage2_minutes", "stage2_enabled", "stage1_unimproved_seconds", "stage1_stop_when_feasible",
    "stage2_unimproved_seconds",
    "random_seed", "allow_cut_fixed_trip")
RECEIVE_KEYS = CONNECTION_KEYS + ("run_info", "run_id", "result_dir", "watch", "download")
ALIASES = {"url": "azure_url", "azure": "azure_url", "jobid": "run_id", "job_id": "run_id", "output_dir": "out_dir"}
TRUE, FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


class ConfigError(api.ApiError):
    """config.txt is missing, unreadable or has a value that cannot be used."""


class Config:
    def __init__(self, values=None, path=None, unknown=()):
        self.values = dict(values or {})
        self.path = Path(path) if path else None
        self.unknown = list(unknown)

    @property
    def base_dir(self):
        return self.path.resolve().parent if self.path else Path.cwd()

    def has(self, key):
        return self.values.get(key, "") != ""

    def text(self, key, default=None):
        return self.values[key] if self.has(key) else default

    def file(self, key):
        """A path from the file, made absolute against the folder of config.txt (None when not set)."""
        value = self.text(key)
        if value is None:
            return None
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else self.base_dir / p)

    def int(self, key):
        value = self.text(key)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"{self._where()}: {key} must be a whole number (got {value!r})") from None

    def bool(self, key):
        value = self.text(key)
        if value is None:
            return None
        if value.lower() in TRUE:
            return True
        if value.lower() in FALSE:
            return False
        raise ConfigError(f"{self._where()}: {key} must be true or false (got {value!r})")

    def _where(self):
        return str(self.path) if self.path else CONFIG_NAME


def parse_text(text, known=None):
    """-> (values, keys that are not in `known`). Raises nothing; checking the values is up to the getters."""
    values, unknown = {}, []
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        eq = line.find("=")
        if eq <= 0:
            continue
        key = re.sub(r"[-\s]+", "_", line[:eq].strip().lower())
        key = ALIASES.get(key, key)
        value = line[eq + 1:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()          # "45   # minutes" -> "45"
        if known is not None and key not in known and key not in unknown:
            unknown.append(key)
        values[key] = value
    return values, unknown


def load(path, default, known):
    """The Config of `path` (the --config option) or, when that is not given, of `default` (the config.txt of the tool).
    A path that was asked for but does not exist is an error; a missing default file just gives an empty Config
    (everything comes from the command line then)."""
    explicit = path is not None
    p = Path(path) if explicit else Path(default)
    if not p.is_file():
        if explicit:
            raise ConfigError(f"config file not found: {p}")
        return Config()
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {p}: {e}") from None
    values, unknown = parse_text(text, known)
    return Config(values, p, unknown)


def _valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc) and "<" not in url and ">" not in url


def connection(cfg, mode=None, api_url=None, record=None):
    """The server to talk to: command line > config.txt > the run record (3-1 only; `record` = its mode and address).

    local   -> local_url (default http://localhost:3001), the MockServer on this PC
    online  -> azure_url, the API controller on Azure (the same one SchedulerWeb uses)
    --api-url replaces the address of whichever mode is used."""
    record = record or {}
    where = str(cfg.path) if cfg.path else CONFIG_NAME
    from_record = not (mode or cfg.has("mode"))                      # nothing else said which server: use the one of the run
    chosen = (mode or cfg.text("mode") or record.get("mode") or "").strip().lower()
    if not chosen:
        raise ConfigError(f"mode is not set. Put  mode=local  or  mode=online  in {where}, or use --mode.")
    if chosen not in api.MODES:
        raise ConfigError(f'unknown mode "{chosen}" (use local or online)')
    key = "local_url" if chosen == "local" else "azure_url"
    url = api_url or cfg.text(key) or (record.get("apiUrl") if from_record else None)
    if not url and chosen == "local":
        url = api.DEFAULT_LOCAL_URL
    if not url:
        raise ConfigError(f"mode is online but azure_url is not set. Put the address of the API in {where} (azure_url=https://...), or use --api-url.")
    if not _valid_url(url):
        hint = " (it still looks like the placeholder)" if "<" in url else ""
        raise ConfigError(f'{key} is not a usable address: "{url}"{hint}. It must start with http:// or https://')
    return api.Connection(chosen, url.rstrip("/"))