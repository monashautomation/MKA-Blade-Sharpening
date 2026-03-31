"""
Standalone Flask Backend for Robot Control Dashboard
Blade grinder system – updated Modbus loop structure:

  PC writes X/Y + status  →  START=1
  Robot moves              →  sets GRINDER_READY=1
  PC detects (camera on)   →  writes new X/Y, sends GRIND_START=1
  Robot grinds, resets     →  GRIND_START=0  (ready for next cut)
  Repeat per tooth
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pymodbus.client import ModbusTcpClient
import PySpin
import cv2
import numpy as np
from scipy import ndimage
from threading import Lock, Thread
import time
from dataclasses import dataclass
from typing import List, Tuple

app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
modbus_client = None
client_connected = False

pyspin_system = None
pyspin_cam = None
pyspin_cam_list = None
camera_lock = Lock()
camera_active = False
camera_thread = None
last_frame = None
camera_config = {
    'frame_rate': 30.0,
    'exposure_time': 10000,
    'gain': 0.0
}

blade_analyzer = None
last_detection_result = None
detection_enabled = False
pixels_per_mm = 86.96
grinder_position_file = 'grinder_position.json'
stored_grinder_tip = None


# ── Grinder position persistence ──────────────────────────────────────────────
def load_grinder_position():
    global stored_grinder_tip
    import os, json
    if os.path.exists(grinder_position_file):
        try:
            with open(grinder_position_file, 'r') as f:
                data = json.load(f)
                stored_grinder_tip = tuple(data['grinder_tip'])
                print(f"✓ Loaded stored grinder position: {stored_grinder_tip}")
        except Exception as e:
            print(f"⚠ Could not load grinder position: {e}")
            stored_grinder_tip = None
    else:
        stored_grinder_tip = None

def save_grinder_position(grinder_tip):
    import json
    from datetime import datetime
    try:
        data = {
            'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
            'timestamp': datetime.now().isoformat()
        }
        with open(grinder_position_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✓ Saved grinder position: {grinder_tip}")
    except Exception as e:
        print(f"⚠ Could not save grinder position: {e}")

load_grinder_position()


# ── Blade detection ───────────────────────────────────────────────────────────
@dataclass
class ToothProfile:
    tooth_id: int
    apex_point: Tuple[int, int]
    top_valley: Tuple[int, int]
    bottom_valley: Tuple[int, int]
    angle: float
    grinding_point: Tuple[int, int]
    height: float
    move_to_grinder: Tuple[float, float]


class SerratedBladeAnalyzer:
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self.teeth_profiles = []
        self.grinder_tip = None
        self.blade_edge_points = None
        self.grinder_edge_points = None

    def preprocess_image(self, blur_kernel=3):
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
        self.binary = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
        return self.binary

    def detect_blade_and_grinder(self, sampling_step=1):
        blade_edge = []
        grinder_points = []
        for y in range(0, self.height, sampling_step):
            row = self.binary[y, :]
            white_pixels = np.where(row > 150)[0]
            # if len(white_pixels) > 0:
            if len(white_pixels) > 10:

                rightmost = white_pixels[white_pixels > self.width // 3 * 2]
                if len(rightmost) > 0:
                    grinder_points.append((rightmost[0], y))
                leftmost_blade = white_pixels[white_pixels < self.width // 2]
                if len(leftmost_blade)>0:
                    blade_edge.append((leftmost_blade[0], y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
            min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
            self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
            min_x = self.grinder_edge_points[min_x_idx, 0]
            tip_points = self.grinder_edge_points[
                np.abs(self.grinder_edge_points[:, 0] - min_x) < 15]
            self.grinder_edge_center = (
                int(np.mean(tip_points[:, 0])), int(np.mean(tip_points[:, 1])))

        return self.blade_edge_points, self.grinder_tip


    def extract_tooth_profiles(self, window_size=20, min_height_px=100):
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return []

        x_coords = self.blade_edge_points[:, 0]
        y_coords = self.blade_edge_points[:, 1]
        x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

        # ── Remove artifact rows (fully white or black) before computing mean ──
        # if self.image is not None:
        #     img_array = np.array(self.image) if not isinstance(self.image, np.ndarray) else self.image
        #     gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if img_array.ndim == 3 else img_array
        #
        #     valid_mask = np.ones(len(y_coords), dtype=bool)
        #     for i, y in enumerate(y_coords.astype(int)):
        #         if 0 <= y < gray.shape[0]:
        #             row = gray[y, :]
        #             # np.all(row == 0) or
        #             if np.all(row == 255):  # fully black or white
        #                 valid_mask[i] = False
        #
        #     mean_x = np.mean(x_smooth[valid_mask]) if valid_mask.any() else np.mean(x_smooth)
        # else:
        #     mean_x = np.mean(x_smooth)
        if self.image is not None:
            img_array = np.array(self.image) if not isinstance(self.image, np.ndarray) else self.image
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if img_array.ndim == 3 else img_array

            valid_mask = np.ones(len(y_coords), dtype=bool)
            for i, y in enumerate(y_coords.astype(int)):
                if 0 <= y < gray.shape[0]:
                    row = gray[y, :]
                    if np.all(row == 255):
                        valid_mask[i] = False

            # IQR outlier removal on top of the white-row mask
            x_valid = x_smooth[valid_mask]
            if len(x_valid) > 4:
                q1, q3 = np.percentile(x_valid, 25), np.percentile(x_valid, 75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_mask = (x_smooth >= lower) & (x_smooth <= upper)
                combined_mask = valid_mask & outlier_mask
            else:
                combined_mask = valid_mask

            mean_x = np.mean(x_smooth[combined_mask]) if combined_mask.any() else np.mean(x_smooth)
        else:
            q1, q3 = np.percentile(x_smooth, 25), np.percentile(x_smooth, 75)
            iqr = q3 - q1
            combined_mask = (x_smooth >= q1 - 1.5 * iqr) & (x_smooth <= q3 + 1.5 * iqr)
            mean_x = np.mean(x_smooth[combined_mask]) if combined_mask.any() else np.mean(x_smooth)

        # ────────────────────────────────────────────────────────────────────────

        peaks, valleys = [], []

        for i in range(window_size, len(x_smooth) - window_size):
            window = x_smooth[i - window_size:i + window_size]
            if x_smooth[i] == np.max(window) and x_smooth[i] > mean_x + 2:
                peaks.append(i)
            elif x_smooth[i] == np.min(window) and x_smooth[i] < mean_x - 2:
                valleys.append(i)
        peaks = self._filter_close_points(peaks, window_size)
        valleys = self._filter_close_points(valleys, window_size)
        # Save mask debug image
        # if self.image is not None:
        #     mask_img = img_array.copy()
        #     for i, y in enumerate(y_coords.astype(int)):
        #         if 0 <= y < mask_img.shape[0]:
        #             x = int(x_coords[i])
        #             if 0 <= x < mask_img.shape[1]:
        #                 color = (0, 255, 0) if valid_mask[i] else (0, 0, 255)  # green=valid, red=invalid
        #                 cv2.circle(mask_img, (x, y), 3, color, -1)
        #     cv2.imwrite("debug_valid_mask.png", mask_img)
        if self.image is not None:
            mask_img = img_array.copy()
            for i, y in enumerate(y_coords.astype(int)):
                if 0 <= y < mask_img.shape[0]:
                    x = int(x_coords[i])
                    if 0 <= x < mask_img.shape[1]:
                        color = (0, 255, 0) if combined_mask[i] else (0, 0, 255)
                        cv2.circle(mask_img, (x, y), 3, color, -1)
            cv2.imwrite("debug_valid_mask.png", mask_img)
        tooth_profiles = []
        for tooth_id, peak_idx in enumerate(peaks, start=1):
            valleys_above = [v for v in valleys if v < peak_idx]
            valleys_below = [v for v in valleys if v > peak_idx]

            if len(valleys_above) == 0 and len(valleys_below) > 0:
                s = [x_smooth[idx] for idx in range(0, min(window_size // 2, peak_idx))]
                sy = [y_coords[idx] for idx in range(0, min(window_size // 2, peak_idx))]
                top_valley = (int(np.mean(s)), int(np.mean(sy))) if s else (int(x_smooth[0]), int(y_coords[0]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
            elif len(valleys_below) == 0 and len(valleys_above) > 0:
                start = peak_idx
                end = min(len(x_smooth) - 1, peak_idx + window_size * 2)
                s = [x_smooth[idx] for idx in range(start, end)]
                sy = [y_coords[idx] for idx in range(start, end)]
                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(np.mean(s)), int(np.mean(sy))) if s else (int(x_smooth[end]), int(y_coords[end]))
            elif len(valleys_above) > 0 and len(valleys_below) > 0:
                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
            else:
                continue

            tooth_point = (int(x_smooth[peak_idx]), int(y_coords[peak_idx]))
            height = abs(tooth_point[0] - ((top_valley[0] + bottom_valley[0]) / 2))
            angle = self._calculate_tooth_angle(top_valley, tooth_point, bottom_valley)
            move_to_grinder = (
                (self.grinder_tip[0] - tooth_point[0], self.grinder_tip[1] - tooth_point[1])
                if self.grinder_tip else (0, 0))

            tooth_profiles.append(ToothProfile(
                tooth_id=tooth_id, apex_point=tooth_point,
                top_valley=top_valley, bottom_valley=bottom_valley,
                angle=angle, grinding_point=tooth_point,
                height=height, move_to_grinder=move_to_grinder))

        return tooth_profiles

    def _filter_close_points(self, points, min_distance):
        if not points:
            return []
        filtered = [points[0]]
        for p in points[1:]:
            if p - filtered[-1] >= min_distance:
                filtered.append(p)
        return filtered

    def _calculate_tooth_angle(self, top_valley, tooth_point, bottom_valley):
        try:
            v1 = np.array(tooth_point) - np.array(top_valley)
            v2 = np.array(bottom_valley) - np.array(tooth_point)
            return float(np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])))
        except:
            return 0.0

    def analyze_frame(self, use_stored_grinder=True):
        global stored_grinder_tip
        try:
            self.preprocess_image()
            self.detect_blade_and_grinder()

            if self.blade_edge_points is None or len(self.blade_edge_points) < 20:
                return None

            if use_stored_grinder and stored_grinder_tip is not None:
                self.grinder_tip = stored_grinder_tip
            elif self.grinder_tip is not None:
                save_grinder_position(self.grinder_tip)

            self.teeth_profiles = self.extract_tooth_profiles()

            if len(self.teeth_profiles) > 0 and self.grinder_tip:
                return self._generate_coordinates()
            return None
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback; traceback.print_exc()
            return None

    def _generate_coordinates(self):
        global stored_grinder_tip, pixels_per_mm

        if len(self.teeth_profiles) < 2:
            return None

        grinder_tip = self.grinder_tip if self.grinder_tip else stored_grinder_tip
        if not grinder_tip:
            return None

        closest_valley = None
        min_distance = float('inf')

        for i in range(len(self.teeth_profiles) - 1):
            ct = self.teeth_profiles[i]
            nt = self.teeth_profiles[i + 1]
            valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
            valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm

            if move_y_mm > 0.5:
                dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
                if dist < min_distance:
                    min_distance = dist
                    closest_valley = {
                        'valley_x': valley_x, 'valley_y': valley_y,
                        'move_x_mm': move_x_mm, 'move_y_mm': move_y_mm,
                        'between_teeth': f"{ct.tooth_id}-{nt.tooth_id}",
                        'distance_mm': dist
                    }

        if not closest_valley:
            return None

        return {
            'valley_id':        closest_valley['between_teeth'],
            'x_mm':             round(float(closest_valley['move_y_mm']), 2),  # swapped
            'y_mm':             round(float(closest_valley['move_x_mm']), 2),  # swapped
            'valley_x_px':      int(closest_valley['valley_x']),
            'valley_y_px':      int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(grinder_tip[0]),
            'grinder_tip_y_px': int(grinder_tip[1]),
            'num_teeth':        int(len(self.teeth_profiles)),
            'distance_mm':      round(float(closest_valley['distance_mm']), 2),
            'status':           1
        }


# ── Modbus client ─────────────────────────────────────────────────────────────
class BladeDataModbusClient:
    """
    Register map:
      128  REG_BAY_ID         PC→Robot  Bay ID
      129  REG_GRINDER_ID     PC→Robot  Grinder ID
      130  REG_ANGLE          PC→Robot  Angle ×10 (e.g. 45.5° → 455)
      131  REG_DEPTH          PC→Robot  Depth ×100 (e.g. 1.25mm → 125)
      132  REG_LENGTH         PC→Robot  Length mm
      133  REG_CONFIG_VERSION PC→Robot  Config version
      134  REG_DETECTION_X    PC→Robot  X offset (×10, uint16 signed)
      135  REG_DETECTION_Y    PC→Robot  Y offset (×10, uint16 signed)
      136  REG_STATUS         PC→Robot  0=no teeth, 1=teeth ok, 2=error
      137  REG_START          PC→Robot  1=start loop, 0=idle
      138  REG_GRINDER_READY  Robot→PC  1=robot at grinder pos (triggers camera)
      139  REG_GRIND_START    PC→Robot  1=start grind; robot resets to 0 after each move
      140  REG_ESTOP          PC→Robot  1=EMERGENCY STOP — halt all motion immediately
    """
    REG_BAY_ID         = 128
    REG_GRINDER_ID     = 129
    REG_ANGLE          = 130
    REG_DEPTH          = 131
    REG_LENGTH         = 132
    REG_CONFIG_VERSION = 133
    REG_DETECTION_X    = 134
    REG_DETECTION_Y    = 135
    REG_STATUS         = 136
    REG_START          = 137
    REG_GRINDER_READY  = 138
    REG_GRIND_START    = 139
    REG_ESTOP          = 140

    STATUS_NO_TEETH   = 0
    STATUS_TEETH_OK   = 1
    STATUS_ERROR      = 2

    def __init__(self, host='172.24.9.20', port=502, unit=1):
        self.host = host
        self.port = port
        self.unit = unit
        self.client = ModbusTcpClient(host, port=port)
        self.connected = False

    def connect(self):
        if self.client.connect():
            self.connected = True
            print(f"✓ Connected to robot at {self.host}:{self.port}")
            return True
        self.connected = False
        print(f"✗ Could not connect to robot at {self.host}:{self.port}")
        return False

    def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
        """Write blade configuration to registers 128–133."""
        if not self.connected:
            return None
        values = [
            int(bay_id),
            int(grinder_id),
            int(angle * 10),
            int(depth * 100),
            int(length),
            int(config_version)
        ]
        result = self.client.write_registers(address=self.REG_BAY_ID, values=values)
        if not result.isError():
            print(f"✓ Config written: Bay={bay_id}, Grinder={grinder_id}, Angle={angle}°, Depth={depth}mm, Length={length}mm")
        else:
            print(f"✗ Failed to write config: {result}")
        return result

    def write_detection(self, x_mm, y_mm, status):
        if not self.connected:
            return None
        x_val = int(x_mm * 10)
        y_val = int(y_mm * 10)
        x_u16 = x_val if x_val >= 0 else 65536 + x_val
        y_u16 = y_val if y_val >= 0 else 65536 + y_val
        values = [x_u16, y_u16, int(status)]
        result = self.client.write_registers(address=self.REG_DETECTION_X, values=values)
        if not result.isError():
            print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm status={status}")
        return result

    def start_loop(self):
        """PC → Robot: begin the grinding loop."""
        if not self.connected:
            return None
        result = self.client.write_register(address=self.REG_START, value=1)
        if not result.isError():
            print("✓ START=1 sent")
        return result

    def stop_loop(self):
        """PC → Robot: abort / stop loop."""
        if not self.connected:
            return None
        result = self.client.write_register(address=self.REG_START, value=0)
        if not result.isError():
            print("✓ START=0 (loop stopped)")
        return result

    def send_grind_start(self):
        """PC → Robot: start grinding this tooth. Robot resets to 0 when done."""
        if not self.connected:
            return None
        result = self.client.write_register(address=self.REG_GRIND_START, value=1)
        if not result.isError():
            print("✓ GRIND_START=1 sent")
        return result

    def emergency_stop(self):
        """
        PC → Robot: EMERGENCY STOP (REG 140 = 1).
        Also zeros START and GRIND_START to ensure the robot halts.
        """
        if not self.connected:
            return None
        # Write ESTOP=1, and simultaneously zero START + GRIND_START for safety
        self.client.write_register(address=self.REG_START,      value=0)
        self.client.write_register(address=self.REG_GRIND_START, value=0)
        result = self.client.write_register(address=self.REG_ESTOP, value=1)
        if not result.isError():
            print("🚨 EMERGENCY STOP SENT (REG 140=1)")
        else:
            print(f"✗ Failed to send E-STOP: {result}")
        return result

    def clear_estop(self):
        """Reset E-STOP register back to 0 (REG 140 = 0)."""
        if not self.connected:
            return None
        result = self.client.write_register(address=self.REG_ESTOP, value=0)
        if not result.isError():
            print("✓ E-STOP cleared (REG 140=0)")
        return result

    def read_grinder_ready(self):
        """Read robot GRINDER_READY flag. Returns bool or None on error."""
        if not self.connected:
            return None
        result = self.client.read_holding_registers(address=self.REG_GRINDER_READY, count=1)
        if result.isError():
            return None
        return bool(result.registers[0])

    def read_grind_start(self):
        """Read GRIND_START flag. Returns False when robot has completed the cut."""
        if not self.connected:
            return None
        result = self.client.read_holding_registers(address=self.REG_GRIND_START, count=1)
        if result.isError():
            return None
        return bool(result.registers[0])

    def read_all_status(self):
        """Read all loop registers (128 config skipped, 134–140) in one request."""
        if not self.connected:
            return None
        result = self.client.read_holding_registers(address=self.REG_DETECTION_X, count=7)
        if result.isError():
            return None
        r = result.registers
        def s16(v): return v if v < 32768 else v - 65536
        return {
            'detection_x_mm': s16(r[0]) / 10.0,
            'detection_y_mm': s16(r[1]) / 10.0,
            'status':         r[2],
            'start':          r[3],
            'grinder_ready':  r[4],
            'grind_start':    r[5],
            'estop':          r[6],
        }

    def close(self):
        self.client.close()
        self.connected = False
        print("✓ Modbus connection closed")


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    try:
        with open('robot_control_dashboard.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        import os
        return f"<h1>Error: robot_control_dashboard.html not found</h1><p>cwd: {os.getcwd()}</p>", 404

@app.route('/api/blade/<blade_id>', methods=['GET'])
def get_blade(blade_id):
    import sqlite3
    try:
        conn = sqlite3.connect("blade_database.sqlite")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM blades WHERE bade_id = ?", (blade_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return jsonify({"success": True, "blade": dict(row)})
        return jsonify({"success": False, "message": f"Blade '{blade_id}' not found"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# ── Connection ──
@app.route('/api/connect', methods=['POST'])
def connect():
    global modbus_client, client_connected
    data = request.json
    host = data.get('host', '172.24.9.20')
    port = int(data.get('port', 502))
    unit = int(data.get('unit', 1))
    try:
        modbus_client = BladeDataModbusClient(host=host, port=port, unit=unit)
        if modbus_client.connect():
            client_connected = True
            return jsonify({'success': True, 'message': f'Connected to robot at {host}:{port}'})
        client_connected = False
        return jsonify({'success': False, 'message': f'Failed to connect to {host}:{port}'}), 500
    except Exception as e:
        client_connected = False
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    global modbus_client, client_connected
    if modbus_client:
        try:
            modbus_client.close()
            modbus_client = None
            client_connected = False
            return jsonify({'success': True, 'message': 'Disconnected'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    return jsonify({'success': True, 'message': 'Already disconnected'})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({'connected': client_connected and modbus_client is not None})

@app.route('/api/configuration', methods=['POST'])
def send_configuration():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    data = request.json
    try:
        result = modbus_client.write_configuration(
            bay_id=int(data.get('bay_id')),
            grinder_id=int(data.get('grinder_id')),
            angle=float(data.get('angle')),
            depth=float(data.get('depth')),
            length=int(data.get('length')),
            config_version=int(data.get('config_version'))
        )
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'Configuration sent (REG 128–133)'})
        return jsonify({'success': False, 'message': 'Failed to send configuration'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── Detection data ──
@app.route('/api/detection', methods=['POST'])
def send_detection():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    data = request.json
    try:
        result = modbus_client.write_detection(
            x_mm=float(data.get('x_mm')),
            y_mm=float(data.get('y_mm')),
            status=int(data.get('status', 1))
        )
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'Detection data sent'})
        return jsonify({'success': False, 'message': 'Failed to send detection'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── Loop control ──
@app.route('/api/start', methods=['POST'])
def start_robot():
    """Start the grinding loop (START=1)."""
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    try:
        result = modbus_client.start_loop()
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'Loop started (START=1)'})
        return jsonify({'success': False, 'message': 'Failed to start loop'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_robot():
    """Stop the grinding loop (START=0)."""
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    try:
        result = modbus_client.stop_loop()
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'Loop stopped (START=0)'})
        return jsonify({'success': False, 'message': 'Failed to stop loop'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/estop', methods=['POST'])
def emergency_stop():
    """
    EMERGENCY STOP — writes REG 140=1, also zeros START and GRIND_START.
    Works even if the loop is not running.
    """
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected to robot'}), 400
    try:
        result = modbus_client.emergency_stop()
        if result and not result.isError():
            return jsonify({'success': True, 'message': '🚨 EMERGENCY STOP — REG 140=1, START=0, GRIND_START=0'})
        return jsonify({'success': False, 'message': 'Failed to send E-STOP'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/estop/clear', methods=['POST'])
def clear_estop():
    """Clear E-STOP flag (REG 140=0) so the robot can be restarted."""
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    try:
        result = modbus_client.clear_estop()
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'E-STOP cleared (REG 140=0)'})
        return jsonify({'success': False, 'message': 'Failed to clear E-STOP'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/grind_start', methods=['POST'])
def grind_start():
    """
    Send GRIND_START=1 after GRINDER_READY detected.
    Optionally accepts detection data in the same call to atomically
    write X/Y and then trigger grinding.
    """
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    data = request.json or {}
    try:
        # Optionally update detection coords in same call
        if 'x_mm' in data and 'y_mm' in data:
            modbus_client.write_detection(
                x_mm=float(data['x_mm']),
                y_mm=float(data['y_mm']),
                status=int(data.get('status', 1))
            )
        result = modbus_client.send_grind_start()
        if result and not result.isError():
            return jsonify({'success': True, 'message': 'GRIND_START=1 sent'})
        return jsonify({'success': False, 'message': 'Failed to send GRIND_START'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/robot_registers', methods=['GET'])
def robot_registers():
    """
    Poll all 6 loop registers in one shot.
    Dashboard uses this to update GRINDER_READY and GRIND_START state.
    """
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    try:
        status = modbus_client.read_all_status()
        if status:
            return jsonify({'success': True, 'registers': status})
        return jsonify({'success': False, 'message': 'Failed to read registers'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── Camera ──
@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame
    try:
        with camera_lock:
            if camera_active:
                return jsonify({'success': True, 'message': 'Camera already running'})
            print("🎥 Initializing FLIR camera...")
            pyspin_system = PySpin.System.GetInstance()
            pyspin_cam_list = pyspin_system.GetCameras()
            if pyspin_cam_list.GetSize() == 0:
                pyspin_cam_list.Clear()
                pyspin_system.ReleaseInstance()
                pyspin_system = pyspin_cam_list = None
                return jsonify({'success': False, 'message': 'No FLIR cameras detected'}), 500
            pyspin_cam = pyspin_cam_list[0]
            pyspin_cam.Init()
            _configure_pyspin_camera()
            pyspin_cam.BeginAcquisition()
            camera_active = True
            camera_thread = Thread(target=_camera_capture_thread, daemon=True)
            camera_thread.start()
            print("✓ FLIR camera started")
            return jsonify({'success': True, 'message': 'FLIR camera started'})
    except PySpin.SpinnakerException as ex:
        return jsonify({'success': False, 'message': f'PySpin error: {ex}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def _configure_pyspin_camera():
    global pyspin_cam, camera_config
    try:
        nodemap = pyspin_cam.GetNodeMap()
        node_acq = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if PySpin.IsWritable(node_acq):
            node_acq.SetIntValue(node_acq.GetEntryByName('Continuous').GetValue())
        node_fr_en = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsWritable(node_fr_en):
            node_fr_en.SetValue(True)
        node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if PySpin.IsWritable(node_fr):
            node_fr.SetValue(min(node_fr.GetMax(), camera_config['frame_rate']))
        node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsWritable(node_exp_auto):
            node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
        node_exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if PySpin.IsWritable(node_exp):
            node_exp.SetValue(min(node_exp.GetMax(), camera_config['exposure_time']))
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsWritable(node_gain_auto):
            node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if PySpin.IsWritable(node_gain):
            node_gain.SetValue(min(node_gain.GetMax(), camera_config['gain']))
    except PySpin.SpinnakerException as ex:
        print(f"⚠ Camera config warning: {ex}")

def _camera_capture_thread():
    global pyspin_cam, camera_active, last_frame
    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
    while camera_active:
        try:
            image_result = pyspin_cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                pf = image_result.GetPixelFormat()
                if pf == PySpin.PixelFormat_BGR8:
                    frame = image_result.GetNDArray()
                elif pf == PySpin.PixelFormat_Mono8:
                    frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
                else:
                    frame = processor.Convert(image_result, PySpin.PixelFormat_BGR8).GetNDArray()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                with camera_lock:
                    last_frame = frame.copy()
                image_result.Release()
        except PySpin.SpinnakerException:
            time.sleep(0.01)
        except Exception as e:
            if camera_active:
                print(f"⚠ Frame error: {e}")
            time.sleep(0.01)

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame
    try:
        camera_active = False
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=2.0)
        with camera_lock:
            if pyspin_cam:
                try:
                    pyspin_cam.EndAcquisition()
                    pyspin_cam.DeInit()
                except: pass
                pyspin_cam = None
            if pyspin_cam_list:
                pyspin_cam_list.Clear()
                pyspin_cam_list = None
            if pyspin_system:
                pyspin_system.ReleaseInstance()
                pyspin_system = None
            last_frame = None
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def draw_detection_overlay(frame, detection_result):
    overlay = frame.copy()
    if not detection_result:
        return overlay
    try:
        gt = (detection_result.get('grinder_tip_x_px'), detection_result.get('grinder_tip_y_px'))
        if gt and gt[0] > 0:
            gti = (int(gt[0]), int(gt[1]))
            cv2.circle(overlay, gti, 12, (0, 255, 255), 3)
            cv2.putText(overlay, "GRINDER", (gti[0]+20, gti[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        vx, vy = detection_result.get('valley_x_px'), detection_result.get('valley_y_px')
        if vx and vy:
            vp = (int(vx), int(vy))
            cv2.circle(overlay, vp, 10, (255, 0, 255), -1)
            cv2.circle(overlay, vp, 12, (255, 255, 255), 2)
            x_mm = detection_result.get('x_mm', 0)
            y_mm = detection_result.get('y_mm', 0)
            cv2.putText(overlay, f"({x_mm:+.1f}, {y_mm:+.1f})mm",
                        (vp[0]-35, vp[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if gt and gt[0] > 0:
                cv2.arrowedLine(overlay, vp, gti, (0, 255, 255), 2, tipLength=0.02)
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (450, 140), (255, 255, 255), 2)
        cv2.putText(overlay, "DETECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, f"Teeth: {detection_result.get('num_teeth',0)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Valley: {detection_result.get('valley_id','N/A')}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(overlay, f"Dist: {detection_result.get('distance_mm',0):.1f}mm", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print(f"Overlay error: {e}")
    return overlay

@app.route('/api/camera/frame')
def get_camera_frame():
    global camera_active, last_frame, last_detection_result
    if not camera_active or last_frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', blank)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    try:
        with camera_lock:
            frame = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        if last_detection_result:
            frame = draw_detection_overlay(frame, last_detection_result)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes() if ret else np.zeros((480, 640, 3), dtype=np.uint8), mimetype='image/jpeg')
    except Exception as e:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', blank)
        return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/api/camera/capture', methods=['POST'])
def capture_frame():
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    try:
        with camera_lock:
            frame = last_frame.copy()
        filename = f'capture_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, frame)
        return jsonify({'success': True, 'message': 'Captured', 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ── Detection endpoints ──
@app.route('/api/detection/analyze', methods=['POST'])
def analyze_current_frame():
    global camera_active, last_frame, last_detection_result
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    try:
        with camera_lock:
            frame_to_process = last_frame.copy()
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        result = analyzer.analyze_frame()
        if result:
            last_detection_result = result
            return jsonify({'success': True, 'detection': result,
                            'message': f"Valley {result.get('valley_id','N/A')}"})
        return jsonify({'success': False, 'message': 'No teeth detected'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/send_auto', methods=['POST'])
def send_auto_detection():
    global camera_active, last_frame, modbus_client
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    try:
        with camera_lock:
            frame_to_process = last_frame.copy()
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        result = analyzer.analyze_frame()
        if not result:
            return jsonify({'success': False, 'message': 'No teeth detected'}), 404
        modbus_result = modbus_client.write_detection(
            x_mm=result['x_mm'], y_mm=result['y_mm'], status=result['status'])
        if modbus_result and not modbus_result.isError():
            return jsonify({'success': True, 'detection': result,
                            'message': f"Valley {result.get('valley_id','N/A')} sent"})
        return jsonify({'success': False, 'message': 'Failed to send to robot'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    return jsonify({'enabled': detection_enabled, 'last_result': last_detection_result,
                    'pixels_per_mm': pixels_per_mm})

@app.route('/api/detection/calibrate', methods=['POST'])
def calibrate_detection():
    global pixels_per_mm
    data = request.json
    new_ppm = data.get('pixels_per_mm')
    if new_ppm and new_ppm > 0:
        pixels_per_mm = float(new_ppm)
        return jsonify({'success': True, 'pixels_per_mm': pixels_per_mm})
    return jsonify({'success': False, 'message': 'Invalid value'}), 400

@app.route('/api/detection/update_grinder', methods=['POST'])
def update_grinder_position():
    global camera_active, last_frame, stored_grinder_tip
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    try:
        with camera_lock:
            frame_to_process = last_frame.copy()
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        analyzer.analyze_frame(use_stored_grinder=False)
        if analyzer.grinder_tip:
            stored_grinder_tip = analyzer.grinder_tip
            save_grinder_position(stored_grinder_tip)
            return jsonify({'success': True,
                            'grinder_tip': {'x': int(stored_grinder_tip[0]),
                                            'y': int(stored_grinder_tip[1])},
                            'message': f'Grinder updated: {stored_grinder_tip}'})
        return jsonify({'success': False, 'message': 'Could not detect grinder'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/grinder_status', methods=['GET'])
def get_grinder_status():
    if stored_grinder_tip:
        return jsonify({'success': True,
                        'grinder_tip': {'x': int(stored_grinder_tip[0]),
                                        'y': int(stored_grinder_tip[1])},
                        'stored': True})
    return jsonify({'success': True, 'grinder_tip': None, 'stored': False})


if __name__ == '__main__':
    import os
    print("=" * 70)
    print("🤖 BLADE GRINDER CONTROL SYSTEM")
    print("   Register map: X=134  Y=135  STATUS=136  START=137")
    print("                 GRINDER_READY=138  GRIND_START=139")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)

#
# """
# Standalone Flask Backend for Robot Control Dashboard
# Blade grinder system – updated Modbus loop structure:
#
#   PC writes X/Y + status  →  START=1
#   Robot moves              →  sets GRINDER_READY=1
#   PC detects (camera on)   →  writes new X/Y, sends GRIND_START=1
#   Robot grinds, resets     →  GRIND_START=0  (ready for next cut)
#   Repeat per tooth
# """
#
# from flask import Flask, request, jsonify, Response
# from flask_cors import CORS
# from pymodbus.client import ModbusTcpClient
# import PySpin
# import cv2
# import numpy as np
# from scipy import ndimage
# from threading import Lock, Thread
# import time
# from dataclasses import dataclass
# from typing import List, Tuple
#
# app = Flask(__name__)
# CORS(app)
#
# # ── Global state ──────────────────────────────────────────────────────────────
# modbus_client = None
# client_connected = False
#
# pyspin_system = None
# pyspin_cam = None
# pyspin_cam_list = None
# camera_lock = Lock()
# camera_active = False
# camera_ready = False   # True once the first real frame has arrived
# camera_thread = None
# last_frame = None
# camera_config = {
#     'frame_rate': 30.0,
#     'exposure_time': 10000,
#     'gain': 0.0
# }
#
# blade_analyzer = None
# last_detection_result = None
# detection_enabled = False
# pixels_per_mm = 86.96
# grinder_position_file = 'grinder_position.json'
# stored_grinder_tip = None
#
#
# # ── Grinder position persistence ──────────────────────────────────────────────
# def load_grinder_position():
#     global stored_grinder_tip
#     import os, json
#     if os.path.exists(grinder_position_file):
#         try:
#             with open(grinder_position_file, 'r') as f:
#                 data = json.load(f)
#                 stored_grinder_tip = tuple(data['grinder_tip'])
#                 print(f"✓ Loaded stored grinder position: {stored_grinder_tip}")
#         except Exception as e:
#             print(f"⚠ Could not load grinder position: {e}")
#             stored_grinder_tip = None
#     else:
#         stored_grinder_tip = None
#
# def save_grinder_position(grinder_tip):
#     import json
#     from datetime import datetime
#     try:
#         data = {
#             'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
#             'timestamp': datetime.now().isoformat()
#         }
#         with open(grinder_position_file, 'w') as f:
#             json.dump(data, f, indent=4)
#         print(f"✓ Saved grinder position: {grinder_tip}")
#     except Exception as e:
#         print(f"⚠ Could not save grinder position: {e}")
#
# load_grinder_position()
#
#
# # ── Blade detection ───────────────────────────────────────────────────────────
# @dataclass
# class ToothProfile:
#     tooth_id: int
#     apex_point: Tuple[int, int]
#     top_valley: Tuple[int, int]
#     bottom_valley: Tuple[int, int]
#     angle: float
#     grinding_point: Tuple[int, int]
#     height: float
#     move_to_grinder: Tuple[float, float]
#
#
# class SerratedBladeAnalyzer:
#     def __init__(self, image):
#         self.image = image
#         self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         self.height, self.width = self.gray.shape
#         self.teeth_profiles = []
#         self.grinder_tip = None
#         self.blade_edge_points = None
#         self.grinder_edge_points = None
#
#     def preprocess_image(self, blur_kernel=3):
#         self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
#         self.binary = cv2.adaptiveThreshold(
#             self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2)
#         return self.binary
#
#     def detect_blade_and_grinder(self, sampling_step=1):
#         blade_edge = []
#         grinder_points = []
#         for y in range(0, self.height, sampling_step):
#             row = self.binary[y, :]
#             white_pixels = np.where(row > 150)[0]
#             if len(white_pixels) > 0:
#                 blade_edge.append((white_pixels[-1], y))
#                 if len(white_pixels) > 10:
#                     rightmost = white_pixels[white_pixels > self.width // 3 * 2]
#                     if len(rightmost) > 0:
#                         grinder_points.append((rightmost[0], y))
#
#         self.blade_edge_points = np.array(blade_edge) if blade_edge else None
#         self.grinder_edge_points = np.array(grinder_points) if grinder_points else None
#
#         if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
#             min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
#             self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
#             min_x = self.grinder_edge_points[min_x_idx, 0]
#             tip_points = self.grinder_edge_points[
#                 np.abs(self.grinder_edge_points[:, 0] - min_x) < 15]
#             self.grinder_edge_center = (
#                 int(np.mean(tip_points[:, 0])), int(np.mean(tip_points[:, 1])))
#
#         return self.blade_edge_points, self.grinder_tip
#
#     def extract_tooth_profiles(self, window_size=10, min_height_px=100):
#         if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
#             return []
#
#         x_coords = self.blade_edge_points[:, 0]
#         y_coords = self.blade_edge_points[:, 1]
#         x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)
#
#         peaks, valleys = [], []
#         mean_x = np.mean(x_smooth)
#
#         for i in range(window_size, len(x_smooth) - window_size):
#             window = x_smooth[i - window_size:i + window_size]
#             if x_smooth[i] == np.max(window) and x_smooth[i] > mean_x + 5:
#                 peaks.append(i)
#             elif x_smooth[i] == np.min(window) and x_smooth[i] < mean_x - 5:
#                 valleys.append(i)
#
#         peaks = self._filter_close_points(peaks, window_size)
#         valleys = self._filter_close_points(valleys, window_size)
#
#         tooth_profiles = []
#         for tooth_id, peak_idx in enumerate(peaks, start=1):
#             valleys_above = [v for v in valleys if v < peak_idx]
#             valleys_below = [v for v in valleys if v > peak_idx]
#
#             if len(valleys_above) == 0 and len(valleys_below) > 0:
#                 s = [x_smooth[idx] for idx in range(0, min(window_size // 2, peak_idx))]
#                 sy = [y_coords[idx] for idx in range(0, min(window_size // 2, peak_idx))]
#                 top_valley = (int(np.mean(s)), int(np.mean(sy))) if s else (int(x_smooth[0]), int(y_coords[0]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
#             elif len(valleys_below) == 0 and len(valleys_above) > 0:
#                 start = peak_idx
#                 end = min(len(x_smooth) - 1, peak_idx + window_size * 2)
#                 s = [x_smooth[idx] for idx in range(start, end)]
#                 sy = [y_coords[idx] for idx in range(start, end)]
#                 top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(np.mean(s)), int(np.mean(sy))) if s else (int(x_smooth[end]), int(y_coords[end]))
#             elif len(valleys_above) > 0 and len(valleys_below) > 0:
#                 top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
#             else:
#                 continue
#
#             tooth_point = (int(x_smooth[peak_idx]), int(y_coords[peak_idx]))
#             height = abs(tooth_point[0] - ((top_valley[0] + bottom_valley[0]) / 2))
#             angle = self._calculate_tooth_angle(top_valley, tooth_point, bottom_valley)
#             move_to_grinder = (
#                 (self.grinder_tip[0] - tooth_point[0], self.grinder_tip[1] - tooth_point[1])
#                 if self.grinder_tip else (0, 0))
#
#             tooth_profiles.append(ToothProfile(
#                 tooth_id=tooth_id, apex_point=tooth_point,
#                 top_valley=top_valley, bottom_valley=bottom_valley,
#                 angle=angle, grinding_point=tooth_point,
#                 height=height, move_to_grinder=move_to_grinder))
#
#         return tooth_profiles
#
#     def _filter_close_points(self, points, min_distance):
#         if not points:
#             return []
#         filtered = [points[0]]
#         for p in points[1:]:
#             if p - filtered[-1] >= min_distance:
#                 filtered.append(p)
#         return filtered
#
#     def _calculate_tooth_angle(self, top_valley, tooth_point, bottom_valley):
#         try:
#             v1 = np.array(tooth_point) - np.array(top_valley)
#             v2 = np.array(bottom_valley) - np.array(tooth_point)
#             return float(np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])))
#         except:
#             return 0.0
#
#     def analyze_frame(self, use_stored_grinder=True):
#         global stored_grinder_tip
#         try:
#             self.preprocess_image()
#             self.detect_blade_and_grinder()
#
#             if self.blade_edge_points is None or len(self.blade_edge_points) < 20:
#                 return None
#
#             if use_stored_grinder and stored_grinder_tip is not None:
#                 self.grinder_tip = stored_grinder_tip
#             elif self.grinder_tip is not None:
#                 save_grinder_position(self.grinder_tip)
#
#             self.teeth_profiles = self.extract_tooth_profiles()
#
#             if len(self.teeth_profiles) > 0 and self.grinder_tip:
#                 return self._generate_coordinates()
#             return None
#         except Exception as e:
#             print(f"Analysis error: {e}")
#             import traceback; traceback.print_exc()
#             return None
#
#     def _generate_coordinates(self):
#         global stored_grinder_tip, pixels_per_mm
#
#         if len(self.teeth_profiles) < 2:
#             return None
#
#         grinder_tip = self.grinder_tip if self.grinder_tip else stored_grinder_tip
#         if not grinder_tip:
#             return None
#
#         closest_valley = None
#         min_distance = float('inf')
#
#         for i in range(len(self.teeth_profiles) - 1):
#             ct = self.teeth_profiles[i]
#             nt = self.teeth_profiles[i + 1]
#             valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
#             valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
#             move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
#             move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
#
#             if move_y_mm > 0.5:
#                 dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
#                 if dist < min_distance:
#                     min_distance = dist
#                     closest_valley = {
#                         'valley_x': valley_x, 'valley_y': valley_y,
#                         'move_x_mm': move_x_mm, 'move_y_mm': move_y_mm,
#                         'between_teeth': f"{ct.tooth_id}-{nt.tooth_id}",
#                         'distance_mm': dist
#                     }
#
#         if not closest_valley:
#             return None
#
#         return {
#             'valley_id':        closest_valley['between_teeth'],
#             'x_mm':             round(float(closest_valley['move_y_mm']), 2),  # swapped
#             'y_mm':             round(float(closest_valley['move_x_mm']), 2),  # swapped
#             'valley_x_px':      int(closest_valley['valley_x']),
#             'valley_y_px':      int(closest_valley['valley_y']),
#             'grinder_tip_x_px': int(grinder_tip[0]),
#             'grinder_tip_y_px': int(grinder_tip[1]),
#             'num_teeth':        int(len(self.teeth_profiles)),
#             'distance_mm':      round(float(closest_valley['distance_mm']), 2),
#             'status':           1
#         }
#
#
# # ── Modbus client ─────────────────────────────────────────────────────────────
# class BladeDataModbusClient:
#     """
#     Register map:
#       128  REG_BAY_ID         PC→Robot  Bay ID
#       129  REG_GRINDER_ID     PC→Robot  Grinder ID
#       130  REG_ANGLE          PC→Robot  Angle ×10 (e.g. 45.5° → 455)
#       131  REG_DEPTH          PC→Robot  Depth ×100 (e.g. 1.25mm → 125)
#       132  REG_LENGTH         PC→Robot  Length mm
#       133  REG_CONFIG_VERSION PC→Robot  Config version
#       134  REG_DETECTION_X    PC→Robot  X offset (×10, uint16 signed)
#       135  REG_DETECTION_Y    PC→Robot  Y offset (×10, uint16 signed)
#       136  REG_STATUS         PC→Robot  0=no teeth, 1=teeth ok, 2=error
#       137  REG_START          PC→Robot  1=start loop, 0=idle
#       138  REG_GRINDER_READY  Robot→PC  1=robot at grinder pos (triggers camera)
#       139  REG_GRIND_START    PC→Robot  1=start grind; robot resets to 0 after each move
#       140  REG_ESTOP          PC→Robot  1=EMERGENCY STOP — halt all motion immediately
#     """
#     REG_BAY_ID         = 128
#     REG_GRINDER_ID     = 129
#     REG_ANGLE          = 130
#     REG_DEPTH          = 131
#     REG_LENGTH         = 132
#     REG_CONFIG_VERSION = 133
#     REG_DETECTION_X    = 134
#     REG_DETECTION_Y    = 135
#     REG_STATUS         = 136
#     REG_START          = 137
#     REG_GRINDER_READY  = 138
#     REG_GRIND_START    = 139
#     REG_ESTOP          = 140
#
#     STATUS_NO_TEETH   = 0
#     STATUS_TEETH_OK   = 1
#     STATUS_ERROR      = 2
#
#     def __init__(self, host='172.24.89.89', port=502, unit=1):
#         self.host = host
#         self.port = port
#         self.unit = unit
#         self.client = ModbusTcpClient(host, port=port)
#         self.connected = False
#
#     def connect(self):
#         if self.client.connect():
#             self.connected = True
#             print(f"✓ Connected to robot at {self.host}:{self.port}")
#             return True
#         self.connected = False
#         print(f"✗ Could not connect to robot at {self.host}:{self.port}")
#         return False
#
#     def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
#         """Write blade configuration to registers 128–133."""
#         if not self.connected:
#             return None
#         values = [
#             int(bay_id),
#             int(grinder_id),
#             int(angle * 10),
#             int(depth * 100),
#             int(length),
#             int(config_version)
#         ]
#         result = self.client.write_registers(address=self.REG_BAY_ID, values=values)
#         if not result.isError():
#             print(f"✓ Config written: Bay={bay_id}, Grinder={grinder_id}, Angle={angle}°, Depth={depth}mm, Length={length}mm")
#         else:
#             print(f"✗ Failed to write config: {result}")
#         return result
#
#     def write_detection(self, x_mm, y_mm, status):
#         if not self.connected:
#             return None
#         x_val = int(x_mm * 10)
#         y_val = int(y_mm * 10)
#         x_u16 = x_val if x_val >= 0 else 65536 + x_val
#         y_u16 = y_val if y_val >= 0 else 65536 + y_val
#         values = [x_u16, y_u16, int(status)]
#         result = self.client.write_registers(address=self.REG_DETECTION_X, values=values)
#         if not result.isError():
#             print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm status={status}")
#         return result
#
#     def start_loop(self):
#         """PC → Robot: begin the grinding loop."""
#         if not self.connected:
#             return None
#         result = self.client.write_register(address=self.REG_START, value=1)
#         if not result.isError():
#             print("✓ START=1 sent")
#         return result
#
#     def stop_loop(self):
#         """PC → Robot: abort / stop loop."""
#         if not self.connected:
#             return None
#         result = self.client.write_register(address=self.REG_START, value=0)
#         if not result.isError():
#             print("✓ START=0 (loop stopped)")
#         return result
#
#     def send_grind_start(self):
#         """PC → Robot: start grinding this tooth. Robot resets to 0 when done."""
#         if not self.connected:
#             return None
#         result = self.client.write_register(address=self.REG_GRIND_START, value=1)
#         if not result.isError():
#             print("✓ GRIND_START=1 sent")
#         return result
#
#     def emergency_stop(self):
#         """
#         PC → Robot: EMERGENCY STOP (REG 140 = 1).
#         Also zeros START and GRIND_START to ensure the robot halts.
#         """
#         if not self.connected:
#             return None
#         # Write ESTOP=1, and simultaneously zero START + GRIND_START for safety
#         self.client.write_register(address=self.REG_START,      value=0)
#         self.client.write_register(address=self.REG_GRIND_START, value=0)
#         result = self.client.write_register(address=self.REG_ESTOP, value=1)
#         if not result.isError():
#             print("🚨 EMERGENCY STOP SENT (REG 140=1)")
#         else:
#             print(f"✗ Failed to send E-STOP: {result}")
#         return result
#
#     def clear_estop(self):
#         """Reset E-STOP register back to 0 (REG 140 = 0)."""
#         if not self.connected:
#             return None
#         result = self.client.write_register(address=self.REG_ESTOP, value=0)
#         if not result.isError():
#             print("✓ E-STOP cleared (REG 140=0)")
#         return result
#
#     def read_grinder_ready(self):
#         """Read robot GRINDER_READY flag. Returns bool or None on error."""
#         if not self.connected:
#             return None
#         result = self.client.read_holding_registers(address=self.REG_GRINDER_READY, count=1)
#         if result.isError():
#             return None
#         return bool(result.registers[0])
#
#     def read_grind_start(self):
#         """Read GRIND_START flag. Returns False when robot has completed the cut."""
#         if not self.connected:
#             return None
#         result = self.client.read_holding_registers(address=self.REG_GRIND_START, count=1)
#         if result.isError():
#             return None
#         return bool(result.registers[0])
#
#     def read_all_status(self):
#         """Read all loop registers (128 config skipped, 134–140) in one request."""
#         if not self.connected:
#             return None
#         result = self.client.read_holding_registers(address=self.REG_DETECTION_X, count=7)
#         if result.isError():
#             return None
#         r = result.registers
#         def s16(v): return v if v < 32768 else v - 65536
#         return {
#             'detection_x_mm': s16(r[0]) / 10.0,
#             'detection_y_mm': s16(r[1]) / 10.0,
#             'status':         r[2],
#             'start':          r[3],
#             'grinder_ready':  r[4],
#             'grind_start':    r[5],
#             'estop':          r[6],
#         }
#
#     def close(self):
#         self.client.close()
#         self.connected = False
#         print("✓ Modbus connection closed")
#
#
# # ── Flask routes ──────────────────────────────────────────────────────────────
#
# @app.route('/')
# def index():
#     try:
#         with open('robot_control_dashboard.html', 'r') as f:
#             return f.read()
#     except FileNotFoundError:
#         import os
#         return f"<h1>Error: robot_control_dashboard.html not found</h1><p>cwd: {os.getcwd()}</p>", 404
#
# @app.route('/api/blade/<blade_id>', methods=['GET'])
# def get_blade(blade_id):
#     import sqlite3
#     try:
#         conn = sqlite3.connect("blade_database.sqlite")
#         conn.row_factory = sqlite3.Row
#         cur = conn.cursor()
#         cur.execute("SELECT * FROM blades WHERE bade_id = ?", (blade_id,))
#         row = cur.fetchone()
#         conn.close()
#         if row:
#             return jsonify({"success": True, "blade": dict(row)})
#         return jsonify({"success": False, "message": f"Blade '{blade_id}' not found"})
#     except Exception as e:
#         return jsonify({"success": False, "message": str(e)})
#
# # ── Connection ──
# @app.route('/api/connect', methods=['POST'])
# def connect():
#     global modbus_client, client_connected
#     data = request.json
#     host = data.get('host', '172.24.89.89')
#     port = int(data.get('port', 502))
#     unit = int(data.get('unit', 1))
#     try:
#         modbus_client = BladeDataModbusClient(host=host, port=port, unit=unit)
#         if modbus_client.connect():
#             client_connected = True
#             return jsonify({'success': True, 'message': f'Connected to robot at {host}:{port}'})
#         client_connected = False
#         return jsonify({'success': False, 'message': f'Failed to connect to {host}:{port}'}), 500
#     except Exception as e:
#         client_connected = False
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/disconnect', methods=['POST'])
# def disconnect():
#     global modbus_client, client_connected
#     if modbus_client:
#         try:
#             modbus_client.close()
#             modbus_client = None
#             client_connected = False
#             return jsonify({'success': True, 'message': 'Disconnected'})
#         except Exception as e:
#             return jsonify({'success': False, 'message': str(e)}), 500
#     return jsonify({'success': True, 'message': 'Already disconnected'})
#
# @app.route('/api/status', methods=['GET'])
# def get_status():
#     return jsonify({'connected': client_connected and modbus_client is not None})
#
# @app.route('/api/configuration', methods=['POST'])
# def send_configuration():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     data = request.json
#     try:
#         result = modbus_client.write_configuration(
#             bay_id=int(data.get('bay_id')),
#             grinder_id=int(data.get('grinder_id')),
#             angle=float(data.get('angle')),
#             depth=float(data.get('depth')),
#             length=int(data.get('length')),
#             config_version=int(data.get('config_version'))
#         )
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'Configuration sent (REG 128–133)'})
#         return jsonify({'success': False, 'message': 'Failed to send configuration'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# # ── Detection data ──
# @app.route('/api/detection', methods=['POST'])
# def send_detection():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     data = request.json
#     try:
#         result = modbus_client.write_detection(
#             x_mm=float(data.get('x_mm')),
#             y_mm=float(data.get('y_mm')),
#             status=int(data.get('status', 1))
#         )
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'Detection data sent'})
#         return jsonify({'success': False, 'message': 'Failed to send detection'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# # ── Loop control ──
# @app.route('/api/start', methods=['POST'])
# def start_robot():
#     """Start the grinding loop (START=1)."""
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     try:
#         result = modbus_client.start_loop()
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'Loop started (START=1)'})
#         return jsonify({'success': False, 'message': 'Failed to start loop'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/stop', methods=['POST'])
# def stop_robot():
#     """Stop the grinding loop (START=0)."""
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     try:
#         result = modbus_client.stop_loop()
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'Loop stopped (START=0)'})
#         return jsonify({'success': False, 'message': 'Failed to stop loop'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/estop', methods=['POST'])
# def emergency_stop():
#     """
#     EMERGENCY STOP — writes REG 140=1, also zeros START and GRIND_START.
#     Works even if the loop is not running.
#     """
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected to robot'}), 400
#     try:
#         result = modbus_client.emergency_stop()
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': '🚨 EMERGENCY STOP — REG 140=1, START=0, GRIND_START=0'})
#         return jsonify({'success': False, 'message': 'Failed to send E-STOP'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/estop/clear', methods=['POST'])
# def clear_estop():
#     """Clear E-STOP flag (REG 140=0) so the robot can be restarted."""
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     try:
#         result = modbus_client.clear_estop()
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'E-STOP cleared (REG 140=0)'})
#         return jsonify({'success': False, 'message': 'Failed to clear E-STOP'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/grind_start', methods=['POST'])
# def grind_start():
#     """
#     Send GRIND_START=1 after GRINDER_READY detected.
#     Optionally accepts detection data in the same call to atomically
#     write X/Y and then trigger grinding.
#     """
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     data = request.json or {}
#     try:
#         # Optionally update detection coords in same call
#         if 'x_mm' in data and 'y_mm' in data:
#             modbus_client.write_detection(
#                 x_mm=float(data['x_mm']),
#                 y_mm=float(data['y_mm']),
#                 status=int(data.get('status', 1))
#             )
#         result = modbus_client.send_grind_start()
#         if result and not result.isError():
#             return jsonify({'success': True, 'message': 'GRIND_START=1 sent'})
#         return jsonify({'success': False, 'message': 'Failed to send GRIND_START'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/robot_registers', methods=['GET'])
# def robot_registers():
#     """
#     Poll all 6 loop registers in one shot.
#     Dashboard uses this to update GRINDER_READY and GRIND_START state.
#     """
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     try:
#         status = modbus_client.read_all_status()
#         if status:
#             return jsonify({'success': True, 'registers': status})
#         return jsonify({'success': False, 'message': 'Failed to read registers'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# # ── Camera ──
# @app.route('/api/camera/start', methods=['POST'])
# def start_camera():
#     global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame
#     try:
#         with camera_lock:
#             if camera_active:
#                 return jsonify({'success': True, 'message': 'Camera already running'})
#             print("🎥 Initializing FLIR camera...")
#             pyspin_system = PySpin.System.GetInstance()
#             pyspin_cam_list = pyspin_system.GetCameras()
#             if pyspin_cam_list.GetSize() == 0:
#                 pyspin_cam_list.Clear()
#                 pyspin_system.ReleaseInstance()
#                 pyspin_system = pyspin_cam_list = None
#                 return jsonify({'success': False, 'message': 'No FLIR cameras detected'}), 500
#             pyspin_cam = pyspin_cam_list[0]
#             pyspin_cam.Init()
#             _configure_pyspin_camera()
#             pyspin_cam.BeginAcquisition()
#             camera_active = True
#             camera_thread = Thread(target=_camera_capture_thread, daemon=True)
#             camera_thread.start()
#             print("✓ FLIR camera started")
#             return jsonify({'success': True, 'message': 'FLIR camera started'})
#     except PySpin.SpinnakerException as ex:
#         return jsonify({'success': False, 'message': f'PySpin error: {ex}'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/camera/ready', methods=['GET'])
# def camera_ready_status():
#     """Returns true once the capture thread has delivered at least one frame."""
#     return jsonify({'ready': camera_ready and last_frame is not None})
#
# def _configure_pyspin_camera():
#     global pyspin_cam, camera_config
#     try:
#         nodemap = pyspin_cam.GetNodeMap()
#         node_acq = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
#         if PySpin.IsWritable(node_acq):
#             node_acq.SetIntValue(node_acq.GetEntryByName('Continuous').GetValue())
#         node_fr_en = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
#         if PySpin.IsWritable(node_fr_en):
#             node_fr_en.SetValue(True)
#         node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
#         if PySpin.IsWritable(node_fr):
#             node_fr.SetValue(min(node_fr.GetMax(), camera_config['frame_rate']))
#         node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
#         if PySpin.IsWritable(node_exp_auto):
#             node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
#         node_exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
#         if PySpin.IsWritable(node_exp):
#             node_exp.SetValue(min(node_exp.GetMax(), camera_config['exposure_time']))
#         node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
#         if PySpin.IsWritable(node_gain_auto):
#             node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
#         node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
#         if PySpin.IsWritable(node_gain):
#             node_gain.SetValue(min(node_gain.GetMax(), camera_config['gain']))
#     except PySpin.SpinnakerException as ex:
#         print(f"⚠ Camera config warning: {ex}")
#
# def _camera_capture_thread():
#     global pyspin_cam, camera_active, last_frame
#     processor = PySpin.ImageProcessor()
#     processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
#     while camera_active:
#         try:
#             image_result = pyspin_cam.GetNextImage(1000)
#             if not image_result.IsIncomplete():
#                 pf = image_result.GetPixelFormat()
#                 if pf == PySpin.PixelFormat_BGR8:
#                     frame = image_result.GetNDArray()
#                 elif pf == PySpin.PixelFormat_Mono8:
#                     frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
#                 else:
#                     frame = processor.Convert(image_result, PySpin.PixelFormat_BGR8).GetNDArray()
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 with camera_lock:
#                     last_frame = frame.copy()
#                     camera_ready = True   # first real frame received
#                 image_result.Release()
#         except PySpin.SpinnakerException:
#             time.sleep(0.01)
#         except Exception as e:
#             if camera_active:
#                 print(f"⚠ Frame error: {e}")
#             time.sleep(0.01)
#
# @app.route('/api/camera/stop', methods=['POST'])
# def stop_camera():
#     global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_ready, camera_thread, last_frame
#     try:
#         camera_active = False
#         camera_ready = False
#         if camera_thread and camera_thread.is_alive():
#             camera_thread.join(timeout=2.0)
#         with camera_lock:
#             if pyspin_cam:
#                 try:
#                     pyspin_cam.EndAcquisition()
#                     pyspin_cam.DeInit()
#                 except: pass
#                 pyspin_cam = None
#             if pyspin_cam_list:
#                 pyspin_cam_list.Clear()
#                 pyspin_cam_list = None
#             if pyspin_system:
#                 pyspin_system.ReleaseInstance()
#                 pyspin_system = None
#             last_frame = None
#         return jsonify({'success': True, 'message': 'Camera stopped'})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# def draw_detection_overlay(frame, detection_result):
#     overlay = frame.copy()
#     if not detection_result:
#         return overlay
#     try:
#         gt = (detection_result.get('grinder_tip_x_px'), detection_result.get('grinder_tip_y_px'))
#         if gt and gt[0] > 0:
#             gti = (int(gt[0]), int(gt[1]))
#             cv2.circle(overlay, gti, 12, (0, 255, 255), 3)
#             cv2.putText(overlay, "GRINDER", (gti[0]+20, gti[1]-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#         vx, vy = detection_result.get('valley_x_px'), detection_result.get('valley_y_px')
#         if vx and vy:
#             vp = (int(vx), int(vy))
#             cv2.circle(overlay, vp, 10, (255, 0, 255), -1)
#             cv2.circle(overlay, vp, 12, (255, 255, 255), 2)
#             x_mm = detection_result.get('x_mm', 0)
#             y_mm = detection_result.get('y_mm', 0)
#             cv2.putText(overlay, f"({x_mm:+.1f}, {y_mm:+.1f})mm",
#                         (vp[0]-35, vp[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#             if gt and gt[0] > 0:
#                 cv2.arrowedLine(overlay, vp, gti, (0, 255, 255), 2, tipLength=0.02)
#         cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
#         cv2.rectangle(overlay, (10, 10), (450, 140), (255, 255, 255), 2)
#         cv2.putText(overlay, "DETECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         cv2.putText(overlay, f"Teeth: {detection_result.get('num_teeth',0)}", (20, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(overlay, f"Valley: {detection_result.get('valley_id','N/A')}", (20, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#         cv2.putText(overlay, f"Dist: {detection_result.get('distance_mm',0):.1f}mm", (20, 130),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     except Exception as e:
#         print(f"Overlay error: {e}")
#     return overlay
#
# @app.route('/api/camera/frame')
# def get_camera_frame():
#     global camera_active, last_frame, last_detection_result
#     if not camera_active or last_frame is None:
#         blank = np.zeros((480, 640, 3), dtype=np.uint8)
#         _, buf = cv2.imencode('.jpg', blank)
#         return Response(buf.tobytes(), mimetype='image/jpeg')
#     try:
#         with camera_lock:
#             frame = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
#         if last_detection_result:
#             frame = draw_detection_overlay(frame, last_detection_result)
#         ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         return Response(buf.tobytes() if ret else np.zeros((480, 640, 3), dtype=np.uint8), mimetype='image/jpeg')
#     except Exception as e:
#         blank = np.zeros((480, 640, 3), dtype=np.uint8)
#         _, buf = cv2.imencode('.jpg', blank)
#         return Response(buf.tobytes(), mimetype='image/jpeg')
#
# @app.route('/api/camera/capture', methods=['POST'])
# def capture_frame():
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     try:
#         with camera_lock:
#             frame = last_frame.copy()
#         filename = f'capture_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
#         cv2.imwrite(filename, frame)
#         return jsonify({'success': True, 'message': 'Captured', 'filename': filename})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# # ── Detection endpoints ──
# @app.route('/api/detection/analyze', methods=['POST'])
# def analyze_current_frame():
#     global camera_active, last_frame, last_detection_result
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     try:
#         with camera_lock:
#             frame_to_process = last_frame.copy()
#         analyzer = SerratedBladeAnalyzer(frame_to_process)
#         result = analyzer.analyze_frame()
#         if result:
#             last_detection_result = result
#             return jsonify({'success': True, 'detection': result,
#                             'message': f"Valley {result.get('valley_id','N/A')}"})
#         return jsonify({'success': False, 'message': 'No teeth detected'}), 404
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/detection/send_auto', methods=['POST'])
# def send_auto_detection():
#     global camera_active, last_frame, modbus_client
#     if not camera_active:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#
#     # Camera may be initialising — wait up to 2s for the first real frame
#     if last_frame is None:
#         deadline = time.time() + 2.0
#         while last_frame is None and time.time() < deadline:
#             time.sleep(0.05)
#         if last_frame is None:
#             return jsonify({'success': False, 'message': 'Camera warming up — no frame yet'}), 503
#
#     try:
#         with camera_lock:
#             frame_to_process = last_frame.copy()
#         analyzer = SerratedBladeAnalyzer(frame_to_process)
#         result = analyzer.analyze_frame()
#         if not result:
#             return jsonify({'success': False, 'message': 'No teeth detected'}), 404
#         modbus_result = modbus_client.write_detection(
#             x_mm=result['x_mm'], y_mm=result['y_mm'], status=result['status'])
#         if modbus_result and not modbus_result.isError():
#             return jsonify({'success': True, 'detection': result,
#                             'message': f"Valley {result.get('valley_id','N/A')} sent"})
#         return jsonify({'success': False, 'message': 'Failed to send to robot'}), 500
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/detection/status', methods=['GET'])
# def get_detection_status():
#     return jsonify({'enabled': detection_enabled, 'last_result': last_detection_result,
#                     'pixels_per_mm': pixels_per_mm})
#
# @app.route('/api/detection/calibrate', methods=['POST'])
# def calibrate_detection():
#     global pixels_per_mm
#     data = request.json
#     new_ppm = data.get('pixels_per_mm')
#     if new_ppm and new_ppm > 0:
#         pixels_per_mm = float(new_ppm)
#         return jsonify({'success': True, 'pixels_per_mm': pixels_per_mm})
#     return jsonify({'success': False, 'message': 'Invalid value'}), 400
#
# @app.route('/api/detection/update_grinder', methods=['POST'])
# def update_grinder_position():
#     global camera_active, last_frame, stored_grinder_tip
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     try:
#         with camera_lock:
#             frame_to_process = last_frame.copy()
#         analyzer = SerratedBladeAnalyzer(frame_to_process)
#         analyzer.analyze_frame(use_stored_grinder=False)
#         if analyzer.grinder_tip:
#             stored_grinder_tip = analyzer.grinder_tip
#             save_grinder_position(stored_grinder_tip)
#             return jsonify({'success': True,
#                             'grinder_tip': {'x': int(stored_grinder_tip[0]),
#                                             'y': int(stored_grinder_tip[1])},
#                             'message': f'Grinder updated: {stored_grinder_tip}'})
#         return jsonify({'success': False, 'message': 'Could not detect grinder'}), 404
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500
#
# @app.route('/api/detection/grinder_status', methods=['GET'])
# def get_grinder_status():
#     if stored_grinder_tip:
#         return jsonify({'success': True,
#                         'grinder_tip': {'x': int(stored_grinder_tip[0]),
#                                         'y': int(stored_grinder_tip[1])},
#                         'stored': True})
#     return jsonify({'success': True, 'grinder_tip': None, 'stored': False})
#
#
# if __name__ == '__main__':
#     import os
#     print("=" * 70)
#     print("🤖 BLADE GRINDER CONTROL SYSTEM")
#     print("   Register map: X=134  Y=135  STATUS=136  START=137")
#     print("                 GRINDER_READY=138  GRIND_START=139")
#     print("=" * 70)
#     app.run(debug=True, host='0.0.0.0', port=5000)