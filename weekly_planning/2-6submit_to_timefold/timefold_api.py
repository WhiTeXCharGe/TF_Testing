"""
timefold_api.py -- the small HTTP client shared by tool 2-6 (submit) and tool 3-1 (receive).  Standard library only.

It speaks the contract of the API controller (and of the local MockServer, which has the same one):
  POST   /runSolver        multipart: env, sched, params (optional), runId (optional)  ->  202 {runId, status}
  GET    /status/<runId>   ->  {status: Submitted | Running | Completed | Failed | Cancelled, stage, progress, error, ...}
  GET    /download/<runId> ->  the result Schedule (only when Completed)
The solver container reads its input under fixed names, so the files are always sent as EnvConfig.yaml, Schedule.yaml and
SolverParams.yaml, whatever the files on disk are called.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import yaml

MODES = ("real", "local")
DEFAULT_MODE = "real"
DEFAULT_LOCAL_URL = "http://localhost:3001"      # the local MockServer, which runs the solver in Docker
RUN_INFO_NAME = "run_info.json"

# Parameters the tool sets unless told otherwise (they go into SolverParams.yaml). Everything else keeps the solver default.
#   stage 2 on, at most 45 + 15 = 60 minutes in total (the solver default would allow 300 + 60 minutes);
#   Fixed trips taken from the schedule workbook are not cut away by the solver.
DEFAULT_PARAMS = {
    "stage1_minutes": 45,
    "stage2_enabled": True,
    "stage2_minutes": 15,
    "allow_cut_fixed_trip": False,
}


class ApiError(Exception):
    """Something the user can act on: wrong URL, server down, rejected input, ..."""


class NotAvailable(ApiError):
    """The requested mode cannot be used yet."""


# ----------------------------------------------------------------------------- mode / url

def resolve_base_url(mode, api_url=None):
    if mode not in MODES:
        raise ApiError(f"unknown mode {mode!r} (use {' or '.join(MODES)})")
    if mode == "real":
        raise NotAvailable("real mode (the Azure API controller) is not available yet. Use --mode local for now.")
    return (api_url or os.environ.get("TF_API_URL_LOCAL") or DEFAULT_LOCAL_URL).rstrip("/")


# ----------------------------------------------------------------------------- http

def encode_multipart(fields, files):
    """fields {name: text}, files [(name, filename, bytes)] -> (body, content type)."""
    boundary = "----tfboundary" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    for name, filename, data in files:
        head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: application/x-yaml\r\n\r\n").encode("utf-8")
        parts.append(head + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request(method, url, body=None, content_type=None, timeout=60):
    req = urllib.request.Request(url, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:                    # the server answered with an error status
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        raise ApiError(f"cannot reach {url} ({reason}). Is the local server running? (MockServer: node server.js)") from None


def _json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}


def submit(base_url, env_path, schedule_path, params_text=None, run_id=None):
    """Uploads the run; returns the answer {runId, status}."""
    files = [("env", "EnvConfig.yaml", Path(env_path).read_bytes()),
             ("sched", "Schedule.yaml", Path(schedule_path).read_bytes())]
    if params_text:
        files.append(("params", "SolverParams.yaml", params_text.encode("utf-8")))
    body, ctype = encode_multipart({"runId": run_id} if run_id else {}, files)
    status, raw = _request("POST", f"{base_url}/runSolver", body, ctype, timeout=300)
    answer = _json(raw)
    if status != 202:
        raise ApiError(f"the server refused the run ({status}): {answer.get('error') or raw[:200]!r}")
    return answer


def get_status(base_url, run_id):
    status, raw = _request("GET", f"{base_url}/status/{run_id}")
    answer = _json(raw)
    if status == 404:
        raise ApiError(f'run "{run_id}" was not found on {base_url}')
    if status != 200:
        raise ApiError(f"status request failed ({status}): {answer.get('error') or raw[:200]!r}")
    return answer


def download(base_url, run_id):
    status, raw = _request("GET", f"{base_url}/download/{run_id}", timeout=300)
    if status != 200:
        raise ApiError(f"download failed ({status}): {_json(raw).get('error') or raw[:200]!r}")
    return raw


# ----------------------------------------------------------------------------- solver parameters

def build_params(params_file=None, **overrides):
    """DEFAULT_PARAMS <- the keys of params_file (if given) <- the overrides that are not None."""
    params = dict(DEFAULT_PARAMS)
    if params_file:
        loaded = yaml.safe_load(Path(params_file).read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ApiError(f"{params_file} must be a YAML map (key: value)")
        params.update(loaded or {})
    params.update({k: v for k, v in overrides.items() if v is not None})
    for key in ("stage1_minutes", "stage2_minutes", "stage1_unimproved_seconds", "stage2_unimproved_seconds"):
        if key in params and (not isinstance(params[key], int) or isinstance(params[key], bool) or params[key] < 1):
            raise ApiError(f"{key} must be a whole number of at least 1 (got {params[key]!r})")
    return params


def params_yaml(params):
    return yaml.safe_dump(params, sort_keys=False, default_flow_style=False)


def time_budget_minutes(params):
    """Longest possible solving time in minutes: stage 1 plus stage 2 (when it is on)."""
    stage1 = params.get("stage1_minutes", 300)
    stage2 = params.get("stage2_minutes", 60) if params.get("stage2_enabled", True) else 0
    return stage1 + stage2


# ----------------------------------------------------------------------------- run record

def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_run_info(folder, info):
    path = Path(folder) / RUN_INFO_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_run_info(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ApiError(f"cannot read the run record {path}: {e}") from None
