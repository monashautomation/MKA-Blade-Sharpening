#!/usr/bin/env python3
"""
Local server for the blade-sharpening flowchart / DRL builder.

- Serves this folder (so flowcharts.html loads at http://localhost:8755/flowcharts.html)
- POST /save-drl     : writes the generated program to the repo as modbus.drl,
                       archiving the previous modbus.drl into _logs/ first.
- POST /save-project : writes the editor's project JSON into _logs/ (gitignored).
- GET  /config       : reports the resolved paths so the UI can show where it saves.

Pure standard library - no pip install. Run:  python server.py
"""
import json
import shutil
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8755

PROGRAM_DIR = Path(__file__).resolve().parent          # ...\Doosan\Program


def find_repo_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.parent.parent                          # fallback


REPO_ROOT = find_repo_root(PROGRAM_DIR)
LOGS_DIR = PROGRAM_DIR / "_logs"                        # diff modbus_*.drl archives (gitignored)
PROJECTS_DIR = PROGRAM_DIR / "Projects"                 # generated blade_project_*.json (gitignored)
DRL_TARGET = PROGRAM_DIR / "modbus.drl"                 # the repo's program file
BASELINE_FILE = PROGRAM_DIR / "baseline.json"           # confirmed-good editor baseline (git-tracked)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROGRAM_DIR), **kwargs)

    def end_headers(self):
        # never cache during active development - always serve the latest file
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        print("  %s - %s" % (self.address_string(), fmt % args))

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/config":
            return self._json(200, {
                "drl_target": rel(DRL_TARGET),
                "logs_dir": rel(LOGS_DIR),
                "projects_dir": rel(PROJECTS_DIR),
                "repo_root": str(REPO_ROOT),
                "baseline": rel(BASELINE_FILE),
                "baseline_exists": BASELINE_FILE.exists(),
            })
        if path == "/baseline":
            if BASELINE_FILE.exists():
                data = BASELINE_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "no baseline set"})
            return
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/save-drl":
                return self._save_drl()
            if path == "/save-project":
                return self._save_project()
            if path == "/set-baseline":
                return self._set_baseline()
        except Exception as exc:                         # noqa: BLE001
            return self._json(500, {"error": str(exc)})
        self._json(404, {"error": "unknown endpoint"})

    def _save_drl(self):
        text = self._body().decode("utf-8")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        archived = None
        if DRL_TARGET.exists():
            archived = LOGS_DIR / ("modbus_%s.drl" % stamp())
            shutil.copy2(DRL_TARGET, archived)
        DRL_TARGET.write_text(text, encoding="utf-8")
        print("  saved DRL -> %s%s" % (
            rel(DRL_TARGET),
            "  (archived previous -> %s)" % rel(archived) if archived else "",
        ))
        self._json(200, {
            "drl": rel(DRL_TARGET),
            "archived": rel(archived) if archived else None,
        })

    def _save_project(self):
        text = self._body().decode("utf-8")
        json.loads(text)                                 # validate it is JSON
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PROJECTS_DIR / ("blade_project_%s.json" % stamp())
        out.write_text(text, encoding="utf-8")
        (PROJECTS_DIR / "blade_project_latest.json").write_text(text, encoding="utf-8")
        print("  saved project -> %s" % rel(out))
        self._json(200, {"project": rel(out)})

    def _set_baseline(self):
        text = self._body().decode("utf-8")
        json.loads(text)                                 # validate it is JSON
        BASELINE_FILE.write_text(text, encoding="utf-8")
        print("  set baseline -> %s" % rel(BASELINE_FILE))
        self._json(200, {"baseline": rel(BASELINE_FILE)})


def main():
    print("Blade-sharpening builder server")
    print("  serving : %s" % PROGRAM_DIR)
    print("  drl out : %s" % DRL_TARGET)
    print("  logs    : %s  (gitignored)" % LOGS_DIR)
    print("  projects: %s  (gitignored)" % PROJECTS_DIR)
    print("  open    : http://localhost:%d/flowcharts.html" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
