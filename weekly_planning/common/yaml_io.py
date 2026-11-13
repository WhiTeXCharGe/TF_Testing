"""
Reading and writing the YAML files of the weekly flow (Schedule, EnvConfig, FlexibleSchedule ...).
"""
from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)
