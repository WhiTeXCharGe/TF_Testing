"""A tiny stand-in for the API controller / MockServer (same contract), for the tests of tools 2-6 and 3-1."""
import email.policy
import json
import threading
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer


class FakeApi:
    def __init__(self):
        self.uploads = []                 # [{"runId", "files": {field: (filename, bytes)}, "fields": {...}}]
        self.statuses = []                # answers of /status, one per call; the last one repeats
        self.result = b"schedule:\n  result: true\n"
        self.refuse_with = None           # e.g. (400, {"error": "..."}) to refuse /runSolver
        self.status_code = 200            # of /status (404 = run not found)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                if outer.refuse_with:
                    return self._send(*outer.refuse_with)
                msg = BytesParser(policy=email.policy.HTTP).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                files, fields = {}, {}
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        files[name] = (part.get_filename(), part.get_payload(decode=True))
                    else:
                        fields[name] = part.get_payload(decode=True).decode("utf-8")
                run_id = fields.get("runId") or "generated_run"
                outer.uploads.append({"runId": run_id, "files": files, "fields": fields})
                self._send(202, {"runId": run_id, "status": "Submitted"})

            def do_GET(self):
                if self.path.startswith("/status/"):
                    if outer.status_code != 200:
                        return self._send(outer.status_code, {"error": "not found"})
                    status = outer.statuses.pop(0) if len(outer.statuses) > 1 else outer.statuses[0]
                    return self._send(200, status)
                if self.path.startswith("/download/"):
                    return self._send(200, outer.result, "application/octet-stream")
                self._send(404, {"error": "unknown path"})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
