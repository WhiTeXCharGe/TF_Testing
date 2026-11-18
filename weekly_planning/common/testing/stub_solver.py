#!/usr/bin/env python3
"""
stub_solver.py  --  a stand-in for the solver server, to test tools 2-6 / 3-1 and 01_StartSchedule / 02_UpdateSchedule WITHOUT Timefold.

    python stub_solver.py [--port 3001] [--seconds 20] [--result <file>] [--fail] [--data-dir <folder>]
    python stub_solver.py --make-run <runId> --result <file> [--data-dir <folder>]     (makes a finished run, then ends)

It speaks the same contract as the local MockServer and the API controller on Azure (see common/timefold_api.py):
    POST /runSolver, GET /status/<runId>, GET /download/<runId>, GET /health
but it does not solve anything. A run goes Submitted -> Running (stage 1, then stage 2) -> Completed in --seconds seconds
(--fail: -> Failed instead). The "result" it hands out is --result (a Schedule.yaml of your choice, e.g. an old solved week)
or, without that option, the Schedule.yaml that was sent. Every run is kept in --data-dir (default: stub_solver_data in the current folder), so the server can be restarted and an old run is still found.

Use it with  mode = local  and  local_url = http://localhost:<port>  in the config of 2-6 and 3-1 (or of plan_assist).
--make-run puts a run that is already Completed into --data-dir without starting the server: with it, 02_UpdateSchedule (3-1) can be
tried on a hand-written run_info.json (its "runId") although 01_StartSchedule never sent anything. Standard library only. Stop it with Ctrl+C.
"""
from __future__ import annotations

import argparse
import email.policy
import json
import re
import socket
import sys
import threading
import time
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Runs:
    """The runs of the stub: input files and the time they were started, under data_dir/<runId>/."""

    def __init__(self, data_dir, seconds, result_file, fail):
        self.dir = Path(data_dir)
        self.seconds = seconds
        self.result_file = Path(result_file) if result_file else None
        self.fail = fail
        self.lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, run_id, files):
        folder = self.dir / run_id
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (folder / name).write_bytes(data)
        (folder / "started").write_text(str(time.time()), encoding="utf-8")

    def known(self, run_id):
        return (self.dir / run_id / "started").is_file()

    def status(self, run_id):
        """What the solver would answer now, from the time that has passed."""
        done = time.time() - float((self.dir / run_id / "started").read_text(encoding="utf-8"))
        total = max(self.seconds, 0.001)
        if done >= total:
            if self.fail:
                return {"status": "Failed", "stage": None, "progress": 0,
                        "error": {"type": "StubError", "message": "the stub solver was started with --fail"}}
            return {"status": "Completed", "stage": None, "progress": 1, "error": None}
        if done < min(2.0, total / 4):
            return {"status": "Submitted", "stage": None, "progress": 0, "error": None}
        fraction = done / total
        return {"status": "Running", "stage": 1 if fraction < 0.75 else 2, "progress": round(fraction, 2), "error": None}

    def result(self, run_id):
        if self.result_file:
            return self.result_file.read_bytes()
        return (self.dir / run_id / "Schedule.yaml").read_bytes()


def make_handler(runs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[stub] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            for start in range(0, len(data), 65536):          # in pieces; then close the sending side before the socket goes away,
                self.wfile.write(data[start:start + 65536])     # or Windows may cut a large answer (a Schedule is several MB)
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_WR)

        def do_POST(self):
            if self.path != "/runSolver":
                return self._send(404, {"error": "unknown path"})
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            msg = BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
            files, fields = {}, {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files[name] = part.get_payload(decode=True)
                else:
                    fields[name] = part.get_payload(decode=True).decode("utf-8")
            if "env" not in files or "sched" not in files:
                return self._send(400, {"error": "Missing required field: env (EnvConfig.yaml) or sched (Schedule.yaml)"})
            run_id = fields.get("runId", "").strip()
            if not RUN_ID.match(run_id):
                run_id = time.strftime("%Y%m%d_%H%M%S") + "_stub"
            stored = {"EnvConfig.yaml": files["env"], "Schedule.yaml": files["sched"]}
            if "params" in files:
                stored["SolverParams.yaml"] = files["params"]
            with runs.lock:
                runs.add(run_id, stored)
            print(f"[stub] run {run_id} accepted: " + ", ".join(f"{k} {len(v)} bytes" for k, v in stored.items()), flush=True)
            self._send(202, {"runId": run_id, "status": "Submitted"})

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            kind, _, run_id = self.path.lstrip("/").partition("/")
            if kind not in ("status", "download") or not RUN_ID.match(run_id):
                return self._send(404, {"error": "unknown path"})
            if not runs.known(run_id):
                return self._send(404, {"error": f'Run "{run_id}" not found'})
            status = runs.status(run_id)
            if kind == "status":
                return self._send(200, status)
            if status["status"] != "Completed":
                return self._send(409, {"error": f'Run "{run_id}" is not completed (status: {status["status"]})'})
            self._send(200, runs.result(run_id), "application/octet-stream")

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="A stand-in for the solver server (no solving), for testing the weekly planning flow.")
    ap.add_argument("--port", type=int, default=3001, help="port to listen on (default 3001, the port of the local MockServer)")
    ap.add_argument("--seconds", type=float, default=20, help="how long a run takes until it is Completed (default 20)")
    ap.add_argument("--result", default=None, help="a Schedule.yaml to hand out as the result (default: the Schedule.yaml that was sent)")
    ap.add_argument("--fail", action="store_true", help="let every run end as Failed")
    ap.add_argument("--make-run", default=None, metavar="RUN_ID", help="create a finished run with this id (needs --result), then end")
    ap.add_argument("--data-dir", default="stub_solver_data", help="where the runs are kept")
    args = ap.parse_args(argv)
    if args.result and not Path(args.result).is_file():
        print(f"Error: result file not found: {args.result}", file=sys.stderr)
        return 1
    runs = Runs(args.data_dir, args.seconds, args.result, args.fail)
    if args.make_run:
        if not RUN_ID.match(args.make_run) or not args.result:
            print("Error: --make-run needs a run id (letters, digits, _ and -) and --result <file>", file=sys.stderr)
            return 1
        runs.add(args.make_run, {"Schedule.yaml": Path(args.result).read_bytes()})
        (runs.dir / args.make_run / "started").write_text("0", encoding="utf-8")          # started long ago: Completed now
        print(f"Run {args.make_run} made in {runs.dir}: Completed, result = {args.result}")
        return 0
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runs))
    print(f"Stub solver on http://localhost:{args.port}  (a run takes {args.seconds:g} s; runs are kept in {runs.dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
