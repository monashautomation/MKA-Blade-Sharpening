#!/usr/bin/env python3
"""
Local/LAN server for the blade-sharpening flowchart / DRL builder.

Two-tier auth (cookie based):
  - VIEW token : required to connect at all (load the page, read config/baseline).
  - EDIT token : additionally required to change code (save .drl, project, config, baseline).
Tokens live in auth.json (gitignored); auto-generated + printed on first run.
Set your own memorable tokens by editing auth.json, then restart.

Endpoints:
  GET  /config /baseline         (view)
  POST /auth /logout             (open)
  POST /save-drl /save-project /set-config /set-baseline   (edit)

Pure standard library - no pip install. Run:  python server.py

SECURITY: HOST='0.0.0.0' exposes this on the LAN. Auth gates access, but it is a
shared-secret over plain http - use only on a trusted network. HOST='127.0.0.1' locks to this PC.
"""
import json
import secrets
import shutil
import socket
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"        # "0.0.0.0" = reachable from other devices; "127.0.0.1" = this machine only
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
BASELINE_FILE = PROGRAM_DIR / "baseline.json"           # confirmed-good editor baseline (git-tracked)
CONFIG_FILE = PROGRAM_DIR / "save_config.json"          # per-machine save paths (gitignored)
AUTH_FILE = PROGRAM_DIR / "auth.json"                   # per-machine access tokens (gitignored)

DEFAULTS = {
    "drl_target":   str(PROGRAM_DIR / "modbus.drl"),
    "logs_dir":     str(PROGRAM_DIR / "_logs"),
    "projects_dir": str(PROGRAM_DIR / "Projects"),
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update({k: v for k, v in json.loads(CONFIG_FILE.read_text()).items() if k in DEFAULTS})
        except Exception:
            pass
    return cfg


def load_auth():
    if AUTH_FILE.exists():
        try:
            a = json.loads(AUTH_FILE.read_text())
            if a.get("view_token") and a.get("edit_token"):
                return a["view_token"], a["edit_token"], False
        except Exception:
            pass
    view = "view-" + secrets.token_urlsafe(6)
    edit = "edit-" + secrets.token_urlsafe(6)
    AUTH_FILE.write_text(json.dumps({"view_token": view, "edit_token": edit}, indent=2), encoding="utf-8")
    return view, edit, True


CONFIG = load_config()
VIEW_TOKEN, EDIT_TOKEN, AUTH_GENERATED = load_auth()


def token_level(tok):
    tok = (tok or "").strip()
    if not tok:
        return None
    if EDIT_TOKEN and tok == EDIT_TOKEN:
        return "edit"
    if VIEW_TOKEN and tok == VIEW_TOKEN:
        return "view"
    return None


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def disp(path_str) -> str:
    p = Path(path_str)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def resolve_path(raw: str) -> Path:
    p = Path(str(raw).strip())
    if not p.is_absolute():
        p = PROGRAM_DIR / p
    return p.resolve()


def config_payload() -> dict:
    return {
        "drl_target":        CONFIG["drl_target"],
        "logs_dir":          CONFIG["logs_dir"],
        "projects_dir":      CONFIG["projects_dir"],
        "drl_target_disp":   disp(CONFIG["drl_target"]),
        "logs_dir_disp":     disp(CONFIG["logs_dir"]),
        "projects_dir_disp": disp(CONFIG["projects_dir"]),
        "custom":            CONFIG_FILE.exists(),
        "repo_root":         str(REPO_ROOT),
        "baseline":          disp(BASELINE_FILE),
        "baseline_exists":   BASELINE_FILE.exists(),
    }


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


LOGIN_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Connect</title>
<style>
body{font-family:'Segoe UI',system-ui,Arial,sans-serif;background:#eef1f5;color:#1f2328;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
.card{background:#fff;border:1px solid #d0d7de;border-radius:14px;padding:30px 34px;box-shadow:0 6px 30px rgba(0,0,0,.08);width:330px}
h1{font-size:18px;margin:0 0 4px}p{color:#57606a;font-size:13px;margin:0 0 18px}
input{width:100%;padding:10px 12px;border:1px solid #d0d7de;border-radius:8px;font-size:14px;font-family:monospace;box-sizing:border-box}
button{width:100%;margin-top:12px;padding:10px;border:0;border-radius:8px;background:#0969da;color:#fff;font-size:14px;font-weight:700;cursor:pointer}
button:hover{background:#0a5fc0}.err{color:#cf222e;font-size:12.5px;margin-top:10px;min-height:15px}
</style></head><body>
<form class="card" id="f">
<h1>Blade-Sharpening Builder</h1>
<p>Enter the access token to connect.</p>
<input id="t" type="password" placeholder="access token" autocomplete="off" autofocus>
<button type="submit">Connect</button>
<div class="err" id="e"></div>
</form>
<script>
document.getElementById('f').addEventListener('submit', async function(ev){
  ev.preventDefault();
  var t=document.getElementById('t').value.trim();
  var r=await fetch('/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:t})});
  if(r.ok){ location.href='/flowcharts.html'; } else { document.getElementById('e').textContent='Invalid token'; }
});
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROGRAM_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        print("  %s - %s" % (self.address_string(), fmt % args))

    # ---- auth helpers ----
    def _cookie_token(self) -> str:
        for part in self.headers.get("Cookie", "").split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                if k == "bld_token":
                    return v
        return ""

    def req_level(self):
        return token_level(self._cookie_token())

    # ---- response helpers ----
    def _json(self, code, payload, cookie=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, code=200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?")[0]
        lv = self.req_level()
        if path == "/config":
            if not lv:
                return self._json(401, {"error": "auth required"})
            payload = config_payload()
            payload["can_edit"] = (lv == "edit")
            return self._json(200, payload)
        if path == "/baseline":
            if not lv:
                return self._json(401, {"error": "auth required"})
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
        # static files require a view-level cookie; otherwise show the login page
        if not lv:
            return self._html(LOGIN_HTML)
        return super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/auth":
                return self._auth()
            if path == "/logout":
                return self._json(200, {"ok": True}, cookie="bld_token=; Path=/; Max-Age=0")
            if self.req_level() != "edit":
                return self._json(403, {"error": "edit token required"})
            if path == "/save-drl":
                return self._save_drl()
            if path == "/save-project":
                return self._save_project()
            if path == "/set-config":
                return self._set_config()
            if path == "/set-baseline":
                return self._set_baseline()
        except Exception as exc:                         # noqa: BLE001
            return self._json(500, {"error": str(exc)})
        self._json(404, {"error": "unknown endpoint"})

    def _auth(self):
        data = json.loads(self._body().decode("utf-8") or "{}")
        tok = str(data.get("token", ""))
        lv = token_level(tok)
        if not lv:
            return self._json(401, {"error": "invalid token"})
        cookie = "bld_token=%s; Path=/; HttpOnly; SameSite=Lax" % tok
        self._json(200, {"ok": True, "level": lv}, cookie=cookie)

    def _save_drl(self):
        text = self._body().decode("utf-8")
        drl = Path(CONFIG["drl_target"])
        logs = Path(CONFIG["logs_dir"])
        drl.parent.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        archived = None
        if drl.exists():
            archived = logs / ("modbus_%s.drl" % stamp())
            shutil.copy2(drl, archived)
        drl.write_text(text, encoding="utf-8")
        print("  saved DRL -> %s%s" % (
            disp(drl), "  (archived -> %s)" % disp(archived) if archived else "",
        ))
        self._json(200, {"drl": disp(drl), "archived": disp(archived) if archived else None})

    def _save_project(self):
        text = self._body().decode("utf-8")
        json.loads(text)
        proj = Path(CONFIG["projects_dir"])
        proj.mkdir(parents=True, exist_ok=True)
        out = proj / ("blade_project_%s.json" % stamp())
        out.write_text(text, encoding="utf-8")
        (proj / "blade_project_latest.json").write_text(text, encoding="utf-8")
        print("  saved project -> %s" % disp(out))
        self._json(200, {"project": disp(out)})

    def _set_config(self):
        data = json.loads(self._body().decode("utf-8"))
        newcfg, errors = dict(CONFIG), []
        for key in ("drl_target", "logs_dir", "projects_dir"):
            if key not in data or not str(data[key]).strip():
                continue
            p = resolve_path(data[key])
            try:
                if key == "drl_target":
                    if p.exists() and p.is_dir():
                        raise ValueError("is a directory, expected a file path")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    target_dir = p.parent
                else:
                    p.mkdir(parents=True, exist_ok=True)
                    target_dir = p
                probe = target_dir / ".write_test"
                probe.write_text("x", encoding="utf-8")
                probe.unlink()
                newcfg[key] = str(p)
            except Exception as exc:                     # noqa: BLE001
                errors.append("%s: %s" % (key, exc))
        if errors:
            return self._json(400, {"error": "; ".join(errors)})
        CONFIG.clear()
        CONFIG.update(newcfg)
        CONFIG_FILE.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
        print("  config updated: %s" % CONFIG)
        payload = config_payload()
        payload["can_edit"] = True
        self._json(200, {"ok": True, "config": payload})

    def _set_baseline(self):
        text = self._body().decode("utf-8")
        json.loads(text)
        BASELINE_FILE.write_text(text, encoding="utf-8")
        print("  set baseline -> %s" % disp(BASELINE_FILE))
        self._json(200, {"baseline": disp(BASELINE_FILE)})


def main():
    ip = lan_ip()
    bar = "-" * 64
    print(bar)
    print("Blade-sharpening builder server   (port %d)" % PORT)
    print(bar)
    print("  serving   : %s" % PROGRAM_DIR)
    print("  .drl out  : %s" % CONFIG["drl_target"])
    print("  archives  : %s" % CONFIG["logs_dir"])
    print("  projects  : %s" % CONFIG["projects_dir"])
    print(bar)
    print("  ACCESS TOKENS (from auth.json - edit that file for custom tokens):")
    print("    connect/view : %s" % VIEW_TOKEN)
    print("    edit         : %s" % EDIT_TOKEN)
    if AUTH_GENERATED:
        print("    (auto-generated on first run)")
    print(bar)
    print("  This computer : http://localhost:%d/flowcharts.html" % PORT)
    if HOST == "0.0.0.0":
        print("  Other devices : http://%s:%d/flowcharts.html   <-- open this elsewhere" % (ip, PORT))
        print("  By hostname   : http://%s:%d/flowcharts.html" % (socket.gethostname(), PORT))
        print(bar)
        print("  NOTE: shared-secret auth over plain http. Trusted networks only.")
        print("        If unreachable, allow inbound TCP %d in Windows Firewall:" % PORT)
        print('        netsh advfirewall firewall add rule name="BladeBuilder %d" '
              "dir=in action=allow protocol=TCP localport=%d" % (PORT, PORT))
    else:
        print("  (HOST=127.0.0.1 - this machine only)")
    print(bar)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
