"""Dashboard backend: bridges browser <-> Doosan Modbus TCP slave.

Run:
    pip install flask pymodbus
    python server.py            # serves http://localhost:5050
    ROBOT_IP=10.0.0.5 python server.py

Browsers can't speak Modbus TCP, so this is the bridge. Every register
the dashboard cares about is declared in REGISTERS below — add a new
toggle by adding a row here and a row in index.html, nothing else.
"""

import os
from flask import Flask, jsonify, request, send_from_directory
from pymodbus.client import ModbusTcpClient

ROBOT_IP = os.environ.get("ROBOT_IP", "172.24.89.89")
ROBOT_PORT = int(os.environ.get("ROBOT_PORT", "502"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "5050"))

# scale: stored_value = real_value * scale.
# signed: True -> two's complement on 16-bit.
# group:  drives layout in index.html.
# editable: False -> dashboard renders read-only unless the override toggle is on.
REGISTERS = {
    # Config inputs
    "BAY_ID":         {"addr": 128, "scale": 1,   "signed": False, "editable": True,  "group": "config"},
    "GRINDER_ID":     {"addr": 129, "scale": 1,   "signed": False, "editable": True,  "group": "config"},
    "ANGLE":          {"addr": 130, "scale": 10,  "signed": False, "editable": True,  "group": "config"},
    "DEPTH":          {"addr": 131, "scale": 100, "signed": False, "editable": True,  "group": "config"},
    "LENGTH":         {"addr": 132, "scale": 1,   "signed": False, "editable": True,  "group": "config"},
    "CONFIG_VERSION": {"addr": 133, "scale": 1,   "signed": False, "editable": True,  "group": "config"},
    "BLADE_COUNT":    {"addr": 144, "scale": 1,   "signed": False, "editable": True,  "group": "config"},

    # Camera detection
    "DETECTION_X":    {"addr": 134, "scale": 100, "signed": True,  "editable": True,  "group": "detection"},
    "DETECTION_Y":    {"addr": 135, "scale": 100, "signed": True,  "editable": True,  "group": "detection"},

    # Feature toggles + mode
    "FEAT_HOMING":    {"addr": 145, "scale": 1, "signed": False, "editable": True, "group": "toggle"},
    "FEAT_PICKUP":    {"addr": 146, "scale": 1, "signed": False, "editable": True, "group": "toggle"},
    "FEAT_LENGTH":    {"addr": 147, "scale": 1, "signed": False, "editable": True, "group": "toggle"},
    "FEAT_GRIND":     {"addr": 148, "scale": 1, "signed": False, "editable": True, "group": "toggle"},
    "FEAT_PUTDOWN":   {"addr": 149, "scale": 1, "signed": False, "editable": True, "group": "toggle"},
    "MODE":           {"addr": 150, "scale": 1, "signed": False, "editable": True, "group": "toggle"},

    # Last-known derived state (written by stages, overridable by user)
    "BLADE_DEPTH":    {"addr": 160, "scale": 100, "signed": True,  "editable": True, "group": "derived"},
    "BLADE_RIGHT_Y":  {"addr": 161, "scale": 10,  "signed": True,  "editable": True, "group": "derived"},
    "BLADE_LENGTH":   {"addr": 162, "scale": 10,  "signed": False, "editable": True, "group": "derived"},
    "BLADE_LEFT_Y":   {"addr": 163, "scale": 10,  "signed": True,  "editable": True, "group": "derived"},

    # Handshake / runtime state — read-only by default in the UI
    "STATUS":             {"addr": 136, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "START":              {"addr": 137, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "GRIND_READY":        {"addr": 138, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "GRIND":              {"addr": 139, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "EMERGENCY":          {"addr": 140, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "CALIBRATION_READY":  {"addr": 141, "scale": 1,  "signed": False, "editable": False, "group": "handshake"},
    "CALIBRATION_ANGLE":  {"addr": 142, "scale": 10, "signed": False, "editable": False, "group": "handshake"},
}


def decode(raw, meta):
    if meta["signed"] and raw >= 32768:
        raw -= 65536
    return raw / meta["scale"] if meta["scale"] != 1 else raw


def encode(val, meta):
    v = int(round(float(val) * meta["scale"]))
    if v > 32767:
        v = 32767
    if v < -32768:
        v = -32768
    if meta["signed"] and v < 0:
        v += 65536
    return v & 0xFFFF


app = Flask(__name__, static_folder=".", static_url_path="")


def _client():
    c = ModbusTcpClient(ROBOT_IP, port=ROBOT_PORT, timeout=2)
    c.connect()
    return c


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/registers", methods=["GET"])
def read_all():
    c = _client()
    try:
        if not c.is_socket_open():
            return jsonify({"ok": False, "error": f"can't reach {ROBOT_IP}:{ROBOT_PORT}"}), 502
        out = {}
        for name, meta in REGISTERS.items():
            r = c.read_holding_registers(address=meta["addr"], count=1)
            if r.isError():
                out[name] = {"value": None, "error": str(r), "meta": meta}
            else:
                out[name] = {"value": decode(r.registers[0], meta), "raw": r.registers[0], "meta": meta}
        return jsonify({"ok": True, "registers": out, "robot_ip": f"{ROBOT_IP}:{ROBOT_PORT}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        c.close()


@app.route("/api/registers", methods=["POST"])
def write_some():
    data = request.get_json(silent=True) or {}
    c = _client()
    try:
        if not c.is_socket_open():
            return jsonify({"ok": False, "error": f"can't reach {ROBOT_IP}:{ROBOT_PORT}"}), 502
        results = {}
        for name, val in data.items():
            if name not in REGISTERS:
                results[name] = {"ok": False, "error": "unknown register"}
                continue
            meta = REGISTERS[name]
            try:
                raw = encode(val, meta)
                r = c.write_register(address=meta["addr"], value=raw)
                results[name] = {"ok": not r.isError(), "raw": raw}
            except Exception as e:
                results[name] = {"ok": False, "error": str(e)}
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        c.close()


if __name__ == "__main__":
    print(f"Dashboard on http://localhost:{HTTP_PORT}  ->  robot {ROBOT_IP}:{ROBOT_PORT}")
    app.run(host="0.0.0.0", port=HTTP_PORT, debug=False)
