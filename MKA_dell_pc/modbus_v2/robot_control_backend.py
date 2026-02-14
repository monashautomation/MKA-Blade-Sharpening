"""
Standalone Flask Backend for Robot Control Dashboard
All-in-one server with embedded HTML and camera feed
Includes automatic blade detection integration with PySpin FLIR camera
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

# Global client instance
modbus_client = None
client_connected = False

# Camera globals (PySpin)
pyspin_system = None
pyspin_cam = None
pyspin_cam_list = None
camera_lock = Lock()
camera_active = False
camera_thread = None
last_frame = None
camera_config = {
    'frame_rate': 30.0,
    'exposure_time': 10000,  # microseconds
    'gain': 0.0
}

# Blade detection globals
blade_analyzer = None
last_detection_result = None
detection_enabled = False
pixels_per_mm = 86.96  # Default calibration
grinder_position_file = 'grinder_position.json'
stored_grinder_tip = None

def load_grinder_position():
    """Load stored grinder position from file"""
    global stored_grinder_tip
    import os
    import json

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
    """Save grinder position to file"""
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

# Load grinder position on startup
load_grinder_position()

@dataclass
class ToothProfile:
    """Data class to store tooth point information"""
    tooth_id: int
    apex_point: Tuple[int, int]
    top_valley: Tuple[int, int]
    bottom_valley: Tuple[int, int]
    angle: float
    grinding_point: Tuple[int, int]
    height: float
    move_to_grinder: Tuple[float, float]

class SerratedBladeAnalyzer:
    """Full analyzer for real-time tooth detection - matches original algorithm"""

    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self.teeth_profiles = []
        self.grinder_tip = None
        self.blade_edge_points = None
        self.grinder_edge_points = None

    def preprocess_image(self, blur_kernel=3):
        """Preprocess image for edge detection"""
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
        self.binary = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return self.binary

    def detect_blade_and_grinder(self, sampling_step=1):
        """Detect blade edge and grinder tip"""
        blade_edge = []
        grinder_points = []

        for y in range(0, self.height, sampling_step):
            row = self.binary[y, :]
            white_pixels = np.where(row > 170)[0]

            if len(white_pixels) > 0:
                blade_edge.append((white_pixels[0], y))

                if len(white_pixels) > 10:
                    rightmost_region = white_pixels[white_pixels > self.width // 3 * 2]
                    if len(rightmost_region) > 0:
                        grinder_points.append((rightmost_region[0], y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        # Find grinder tip (leftmost point)
        if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
            min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
            self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])

            # Average nearby points for accuracy
            min_x = self.grinder_edge_points[min_x_idx, 0]
            tip_points = self.grinder_edge_points[
                np.abs(self.grinder_edge_points[:, 0] - min_x) < 15
            ]

            self.grinder_edge_center = (
                int(np.mean(tip_points[:, 0])),
                int(np.mean(tip_points[:, 1]))
            )

        return self.blade_edge_points, self.grinder_tip

    def extract_tooth_profiles(self, window_size=20, min_height_px=100):
        """Extract tooth POINTS (peaks) from blade edge"""
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return []

        x_coords = self.blade_edge_points[:, 0]
        y_coords = self.blade_edge_points[:, 1]

        # Smooth x-coordinates
        x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

        # Find peaks (TOOTH TIPS) and valleys (grooves)
        peaks = []
        valleys = []

        mean_x = np.mean(x_smooth)

        for i in range(window_size, len(x_smooth) - window_size):
            window = x_smooth[i - window_size:i + window_size]

            # PEAKS are tooth tips (pointing right = higher x values)
            if x_smooth[i] == np.max(window) and x_smooth[i] > mean_x + 10:
                peaks.append(i)
            # Valleys are grooves between teeth
            elif x_smooth[i] == np.min(window) and x_smooth[i] < mean_x - 10:
                valleys.append(i)

        # Filter close points
        peaks = self._filter_close_points(peaks, window_size)
        valleys = self._filter_close_points(valleys, window_size)

        # Create tooth profiles
        tooth_profiles = []
        tooth_id = 1

        for peak_idx in peaks:
            # Find valleys above and below this peak
            valleys_above = [v for v in valleys if v < peak_idx]
            valleys_below = [v for v in valleys if v > peak_idx]

            # Handle edge cases
            if len(valleys_above) == 0 and len(valleys_below) > 0:
                sample_size = min(window_size // 2, 50)
                sample_end = min(sample_size, peak_idx)
                sample_indices = range(0, sample_end)

                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                top_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (int(x_smooth[0]), int(y_coords[0]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))

            elif len(valleys_below) == 0 and len(valleys_above) > 0:
                sample_start = peak_idx
                sample_end = min(len(x_smooth) - 1, peak_idx + window_size * 2)
                sample_indices = range(sample_start, sample_end)

                sampled_x = [x_smooth[idx] for idx in sample_indices if idx < len(x_smooth)]
                sampled_y = [y_coords[idx] for idx in sample_indices if idx < len(y_coords)]

                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(np.mean(sampled_x)), int(np.mean(sampled_y))) if sampled_x else (int(x_smooth[sample_end]), int(y_coords[sample_end]))

            elif len(valleys_above) > 0 and len(valleys_below) > 0:
                top_valley = (int(x_smooth[valleys_above[-1]]), int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
            else:
                continue

            tooth_point = (int(x_smooth[peak_idx]), int(y_coords[peak_idx]))

            # Calculate tooth height
            height = abs(tooth_point[0] - ((top_valley[0] + bottom_valley[0]) / 2))

            # Calculate angle
            angle = self._calculate_tooth_angle(top_valley, tooth_point, bottom_valley)

            # Calculate movement to grinder
            if self.grinder_tip is not None:
                move_to_grinder = (
                    self.grinder_tip[0] - tooth_point[0],
                    self.grinder_tip[1] - tooth_point[1]
                )
            else:
                move_to_grinder = (0, 0)

            tooth_profiles.append(ToothProfile(
                tooth_id=tooth_id,
                apex_point=tooth_point,
                top_valley=top_valley,
                bottom_valley=bottom_valley,
                angle=angle,
                grinding_point=tooth_point,
                height=height,
                move_to_grinder=move_to_grinder
            ))

            tooth_id += 1

        return tooth_profiles

    def _filter_close_points(self, points, min_distance):
        """Filter out points too close together"""
        if len(points) == 0:
            return []

        filtered = [points[0]]
        for point in points[1:]:
            if point - filtered[-1] >= min_distance:
                filtered.append(point)
        return filtered

    def _calculate_tooth_angle(self, top_valley, tooth_point, bottom_valley):
        """Calculate tooth angle"""
        try:
            vec1 = np.array(tooth_point) - np.array(top_valley)
            vec2 = np.array(bottom_valley) - np.array(tooth_point)

            angle1 = np.arctan2(vec1[1], vec1[0])
            angle2 = np.arctan2(vec2[1], vec2[0])

            angle_diff = np.degrees(angle2 - angle1)
            return float(angle_diff)
        except:
            return 0.0

    def analyze_frame(self, use_stored_grinder=True):
        """Complete analysis pipeline for real-time detection"""
        global stored_grinder_tip

        try:
            # Preprocess
            self.preprocess_image()

            # Detect blade and grinder
            self.detect_blade_and_grinder()

            if self.blade_edge_points is None or len(self.blade_edge_points) < 50:
                return None

            # Use stored grinder position if available and requested
            if use_stored_grinder and stored_grinder_tip is not None:
                self.grinder_tip = stored_grinder_tip
                print(f"  Using stored grinder tip: {stored_grinder_tip}")
            elif self.grinder_tip is not None:
                # Save newly detected grinder position
                save_grinder_position(self.grinder_tip)
                print(f"  Detected and saved new grinder tip: {self.grinder_tip}")

            # Extract teeth
            self.teeth_profiles = self.extract_tooth_profiles()

            # Generate coordinates for first tooth
            if len(self.teeth_profiles) > 0 and self.grinder_tip:
                return self._generate_coordinates()

            return None

        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_coordinates(self):
        """
        Generate coordinates for closest valley (groove) above grinder
        Matches TCP server logic exactly:
        - Finds valley (middle point between teeth)
        - Must be above grinder (move_y_mm > 1)
        - Sends closest valley
        - X and Y are SWAPPED: x_mm = move_y_mm, y_mm = move_x_mm
        """
        global stored_grinder_tip, pixels_per_mm

        if len(self.teeth_profiles) < 2:  # Need at least 2 teeth to have a valley
            return None

        grinder_tip = self.grinder_tip if self.grinder_tip else stored_grinder_tip
        if not grinder_tip:
            return None

        # Find closest valley (groove between teeth) that is above grinder
        closest_valley = None
        min_distance = float('inf')

        for i in range(len(self.teeth_profiles) - 1):
            current_tooth = self.teeth_profiles[i]
            next_tooth = self.teeth_profiles[i + 1]

            # Calculate middle point between current tooth tip and next tooth tip
            # This is the valley/groove between the teeth
            valley_x = (current_tooth.grinding_point[0] + next_tooth.grinding_point[0]) / 2
            valley_y = (current_tooth.grinding_point[1] + next_tooth.grinding_point[1]) / 2

            # Calculate movement to grinder from valley
            move_x_px = grinder_tip[0] - valley_x
            move_y_px = grinder_tip[1] - valley_y

            move_x_mm = move_x_px / pixels_per_mm
            move_y_mm = move_y_px / pixels_per_mm

            # Check if valley is above grinder (positive Y offset > 1mm)
            if move_y_mm > 1:
                distance_mm = ((move_x_mm ** 2 + move_y_mm ** 2) ** 0.5)

                if distance_mm < min_distance:
                    min_distance = distance_mm
                    closest_valley = {
                        'valley_x': valley_x,
                        'valley_y': valley_y,
                        'move_x_mm': move_x_mm,
                        'move_y_mm': move_y_mm,
                        'between_teeth': f"{current_tooth.tooth_id}-{next_tooth.tooth_id}",
                        'distance_mm': distance_mm
                    }

        if not closest_valley:
            return None

        # IMPORTANT: X and Y are SWAPPED to match TCP server protocol
        # x_mm = move_y_mm (vertical movement)
        # y_mm = move_x_mm (horizontal movement)
        return {
            'valley_id': closest_valley['between_teeth'],
            'x_mm': round(float(closest_valley['move_y_mm']), 2),  # SWAPPED
            'y_mm': round(float(closest_valley['move_x_mm']), 2),  # SWAPPED
            'valley_x_px': int(closest_valley['valley_x']),
            'valley_y_px': int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(grinder_tip[0]),
            'grinder_tip_y_px': int(grinder_tip[1]),
            'num_teeth': int(len(self.teeth_profiles)),
            'distance_mm': round(float(closest_valley['distance_mm']), 2),
            'status': 1  # Valid detection
        }

class BladeDataModbusClient:
    """Modbus TCP client to write blade data to robot"""

    # Robot register mapping
    REG_BAY_ID = 128
    REG_GRINDER_ID = 129
    REG_ANGLE = 130
    REG_DEPTH = 131
    REG_LENGTH = 132
    REG_CONFIG_VERSION = 133
    REG_DETECTION_X = 134
    REG_DETECTION_Y = 135
    REG_STATUS = 136
    REG_COMMAND = 137
    REG_START = 138

    # Command codes
    CMD_READ_DETECTION = 11
    CMD_START_GRINDING = 20
    CMD_STOP = 21
    CMD_RESET = 22

    def __init__(self, host='172.24.89.89', port=502, unit=1):
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
        else:
            self.connected = False
            print(f"✗ Could not connect to robot at {self.host}:{self.port}")
            return False

    def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
        if not self.connected:
            print("✗ Not connected to robot")
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
            print(f"✓ Configuration written successfully")
        return result

    def write_detection(self, x_mm, y_mm, status):
        if not self.connected:
            return None

        x_val = int(x_mm * 10)
        y_val = int(y_mm * 10)
        x_unsigned = x_val if x_val >= 0 else 65536 + x_val
        y_unsigned = y_val if y_val >= 0 else 65536 + y_val
        status_int = int(status)
        values = [x_unsigned, y_unsigned, status_int]
        result = self.client.write_registers(address=self.REG_DETECTION_X, values=values)

        if not result.isError():
            print(f"✓ Detection data written: X={x_mm:.2f}mm, Y={y_mm:.2f}mm")
        return result

    def write_command(self, command):
        if not self.connected:
            return None
        result = self.client.write_register(address=self.REG_COMMAND, value=command)
        if not result.isError():
            print(f"✓ Command written: {command}")
        return result

    def start_robot(self):
        if not self.connected:
            return None
        print("\n🚀 Starting robot operation...")
        result = self.client.write_register(address=self.REG_START, value=1)
        if not result.isError():
            print(f"✓ Robot started successfully!")
        return result

    def stop_robot(self):
        if not self.connected:
            return None
        print("\n🛑 Stopping robot...")
        result = self.client.write_register(address=self.REG_START, value=0)
        if not result.isError():
            print(f"✓ Robot stopped")
        return result

    def close(self):
        self.client.close()
        self.connected = False
        print("✓ Modbus client connection closed")


@app.route('/')
def index():
    """Serve the dashboard HTML"""
    html_file = 'robot_control_dashboard.html'
    try:
        with open(html_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <body>
            <h1>Error: Dashboard file not found</h1>
            <p>Please ensure 'robot_control_dashboard.html' is in the same directory as this script.</p>
            <p>Current directory: """ + os.getcwd() + """</p>
        </body>
        </html>
        """, 404

@app.route('/api/connect', methods=['POST'])
def connect():
    global modbus_client, client_connected

    data = request.json
    host = data.get('host', '172.24.89.89')
    port = int(data.get('port', 502))
    unit = int(data.get('unit', 1))

    try:
        modbus_client = BladeDataModbusClient(host=host, port=port, unit=unit)
        if modbus_client.connect():
            client_connected = True
            return jsonify({
                'success': True,
                'message': f'Connected to robot at {host}:{port}'
            })
        else:
            client_connected = False
            return jsonify({
                'success': False,
                'message': f'Failed to connect to robot at {host}:{port}'
            }), 500
    except Exception as e:
        client_connected = False
        return jsonify({
            'success': False,
            'message': f'Connection error: {str(e)}'
        }), 500

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    global modbus_client, client_connected

    if modbus_client:
        try:
            modbus_client.close()
            modbus_client = None
            client_connected = False
            return jsonify({
                'success': True,
                'message': 'Disconnected from robot'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Disconnect error: {str(e)}'
            }), 500

    return jsonify({
        'success': True,
        'message': 'Already disconnected'
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    global modbus_client, client_connected
    return jsonify({
        'connected': client_connected and modbus_client is not None
    })

@app.route('/api/configuration', methods=['POST'])
def send_configuration():
    global modbus_client

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

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
            return jsonify({
                'success': True,
                'message': 'Configuration sent successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send configuration'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Configuration error: {str(e)}'
        }), 500

@app.route('/api/detection', methods=['POST'])
def send_detection():
    global modbus_client

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

    data = request.json

    try:
        result = modbus_client.write_detection(
            x_mm=float(data.get('x_mm')),
            y_mm=float(data.get('y_mm')),
            status=int(data.get('status'))
        )

        if result and not result.isError():
            return jsonify({
                'success': True,
                'message': 'Detection data sent successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send detection data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Detection error: {str(e)}'
        }), 500

@app.route('/api/command', methods=['POST'])
def send_command():
    global modbus_client

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

    data = request.json

    try:
        command = int(data.get('command'))
        result = modbus_client.write_command(command)

        if result and not result.isError():
            return jsonify({
                'success': True,
                'message': f'Command {command} sent successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to send command {command}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Command error: {str(e)}'
        }), 500

@app.route('/api/start', methods=['POST'])
def start_robot():
    global modbus_client

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

    try:
        result = modbus_client.start_robot()

        if result and not result.isError():
            return jsonify({
                'success': True,
                'message': 'Robot started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start robot'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Start error: {str(e)}'
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_robot():
    global modbus_client

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

    try:
        result = modbus_client.stop_robot()

        if result and not result.isError():
            return jsonify({
                'success': True,
                'message': 'Robot stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to stop robot'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Stop error: {str(e)}'
        }), 500

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame

    try:
        with camera_lock:
            if camera_active:
                return jsonify({
                    'success': True,
                    'message': 'Camera already running'
                })

            # Initialize PySpin system
            print("🎥 Initializing FLIR camera with PySpin...")
            pyspin_system = PySpin.System.GetInstance()
            pyspin_cam_list = pyspin_system.GetCameras()

            if pyspin_cam_list.GetSize() == 0:
                print("✗ No FLIR cameras detected!")
                pyspin_cam_list.Clear()
                pyspin_system.ReleaseInstance()
                pyspin_system = None
                pyspin_cam_list = None
                return jsonify({
                    'success': False,
                    'message': 'No FLIR cameras detected. Check camera connection.'
                }), 500

            # Get first camera
            pyspin_cam = pyspin_cam_list[0]
            pyspin_cam.Init()

            # Configure camera
            _configure_pyspin_camera()

            # Start acquisition
            pyspin_cam.BeginAcquisition()
            camera_active = True

            # Start capture thread
            camera_thread = Thread(target=_camera_capture_thread, daemon=True)
            camera_thread.start()

            print("✓ FLIR camera started successfully")

            return jsonify({
                'success': True,
                'message': 'FLIR camera started successfully'
            })

    except PySpin.SpinnakerException as ex:
        print(f"✗ PySpin error: {str(ex)}")
        return jsonify({
            'success': False,
            'message': f'PySpin camera error: {str(ex)}'
        }), 500
    except Exception as e:
        print(f"✗ Camera start error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Camera error: {str(e)}'
        }), 500

def _configure_pyspin_camera():
    """Configure FLIR camera settings"""
    global pyspin_cam, camera_config

    try:
        nodemap = pyspin_cam.GetNodeMap()

        # Set continuous acquisition mode
        node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if PySpin.IsWritable(node_acq_mode):
            node_continuous = node_acq_mode.GetEntryByName('Continuous')
            node_acq_mode.SetIntValue(node_continuous.GetValue())
            print("  ✓ Set to continuous acquisition mode")

        # Enable and set frame rate
        node_fr_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsWritable(node_fr_enable):
            node_fr_enable.SetValue(True)

        node_fr = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if PySpin.IsWritable(node_fr):
            max_fr = node_fr.GetMax()
            target_fr = min(max_fr, camera_config['frame_rate'])
            node_fr.SetValue(target_fr)
            print(f"  ✓ Frame rate set to {target_fr} fps")

        # Set exposure time
        node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsWritable(node_exp_auto):
            node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())

        node_exp = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if PySpin.IsWritable(node_exp):
            max_exp = node_exp.GetMax()
            target_exp = min(max_exp, camera_config['exposure_time'])
            node_exp.SetValue(target_exp)
            print(f"  ✓ Exposure time set to {target_exp} µs")

        # Set gain
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsWritable(node_gain_auto):
            node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())

        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if PySpin.IsWritable(node_gain):
            max_gain = node_gain.GetMax()
            target_gain = min(max_gain, camera_config['gain'])
            node_gain.SetValue(target_gain)
            print(f"  ✓ Gain set to {target_gain}")

    except PySpin.SpinnakerException as ex:
        print(f"⚠ Warning during camera configuration: {ex}")

def _camera_capture_thread():
    """Thread for continuously capturing frames from FLIR camera"""
    global pyspin_cam, camera_active, last_frame

    print("📹 Camera capture thread started")

    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    while camera_active:
        try:
            # Get next image with 1000ms timeout
            image_result = pyspin_cam.GetNextImage(1000)

            if not image_result.IsIncomplete():
                pixel_format = image_result.GetPixelFormat()

                # Convert to BGR format
                if pixel_format == PySpin.PixelFormat_BGR8:
                    frame = image_result.GetNDArray()
                elif pixel_format == PySpin.PixelFormat_Mono8:
                    frame = cv2.cvtColor(image_result.GetNDArray(), cv2.COLOR_GRAY2BGR)
                else:
                    # Convert other formats to BGR
                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                    frame = image_converted.GetNDArray()

                # Rotate 90 degrees counter-clockwise (as in original code)
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                with camera_lock:
                    last_frame = frame.copy()

                image_result.Release()

        except PySpin.SpinnakerException:
            time.sleep(0.01)
        except Exception as e:
            if camera_active:
                print(f"⚠ Frame capture error: {e}")
            time.sleep(0.01)

    print("📹 Camera capture thread stopped")

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame

    try:
        camera_active = False

        # Wait for capture thread to stop
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=2.0)

        with camera_lock:
            # Stop acquisition
            if pyspin_cam is not None:
                try:
                    pyspin_cam.EndAcquisition()
                    pyspin_cam.DeInit()
                except:
                    pass
                pyspin_cam = None

            # Clear camera list
            if pyspin_cam_list is not None:
                pyspin_cam_list.Clear()
                pyspin_cam_list = None

            # Release system
            if pyspin_system is not None:
                pyspin_system.ReleaseInstance()
                pyspin_system = None

            last_frame = None

        print("✓ FLIR camera stopped")

        return jsonify({
            'success': True,
            'message': 'Camera stopped'
        })
    except Exception as e:
        print(f"⚠ Camera stop error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Camera stop error: {str(e)}'
        }), 500

def draw_detection_overlay(frame, detection_result):
    """Draw detection overlay on frame matching original visualization"""
    overlay = frame.copy()

    if not detection_result:
        return overlay

    try:
        grinder_tip = (detection_result.get('grinder_tip_x_px'), detection_result.get('grinder_tip_y_px'))

        # Draw grinder tip
        if grinder_tip and grinder_tip[0] > 0:
            grinder_tip_int = (int(grinder_tip[0]), int(grinder_tip[1]))
            cv2.circle(overlay, grinder_tip_int, 12, (0, 255, 255), 3)
            cv2.circle(overlay, grinder_tip_int, 15, (0, 0, 0), 2)

            label = "GRINDER"
            cv2.putText(overlay, label, (grinder_tip_int[0] + 20, grinder_tip_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw valley position
        valley_x = detection_result.get('valley_x_px')
        valley_y = detection_result.get('valley_y_px')

        if valley_x and valley_y:
            valley_pos = (int(valley_x), int(valley_y))

            # Draw valley marker (different color from tooth)
            cv2.circle(overlay, valley_pos, 10, (255, 0, 255), -1)  # Magenta
            cv2.circle(overlay, valley_pos, 12, (255, 255, 255), 2)

            valley_id = detection_result.get('valley_id', 'N/A')
            cv2.putText(overlay, f"V {valley_id}", (valley_pos[0] - 30, valley_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Draw coordinates box
            x_mm = detection_result.get('x_mm', 0)
            y_mm = detection_result.get('y_mm', 0)
            offset_text = f"({x_mm:+.1f}, {y_mm:+.1f})mm"
            text_size = cv2.getTextSize(offset_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            box_x, box_y = valley_pos[0] - 35, valley_pos[1] - 40

            cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                         (box_x + text_size[0] + 5, box_y + 5), (0, 0, 0), -1)
            cv2.rectangle(overlay, (box_x - 5, box_y - text_size[1] - 5),
                         (box_x + text_size[0] + 5, box_y + 5), (255, 0, 255), 2)
            cv2.putText(overlay, offset_text, (box_x, box_y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Draw arrow to grinder
            if grinder_tip and grinder_tip[0] > 0:
                cv2.arrowedLine(overlay, valley_pos, grinder_tip_int, (0, 255, 255), 2, tipLength=0.02)

        # Info panel
        num_teeth = detection_result.get('num_teeth', 0)
        distance = detection_result.get('distance_mm', 0)

        panel_height = 150
        cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (450, panel_height), (255, 255, 255), 2)

        y = 35
        cv2.putText(overlay, "DETECTION STATUS", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 35

        cv2.putText(overlay, f"Teeth Detected: {num_teeth}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30

        cv2.putText(overlay, f"Valley: {detection_result.get('valley_id', 'N/A')}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y += 30

        cv2.putText(overlay, f"Distance: {distance:.1f}mm", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        print(f"Overlay error: {e}")

    return overlay

@app.route('/api/camera/frame')
def get_camera_frame():
    global camera_active, last_frame, last_detection_result

    if not camera_active or last_frame is None:
        # Return a blank frame
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    try:
        with camera_lock:
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                # Blank frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw overlay if we have detection results
        if last_detection_result:
            frame = draw_detection_overlay(frame, last_detection_result)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
            return Response(buffer.tobytes(), mimetype='image/jpeg')

    except Exception as e:
        print(f"Frame encode error: {str(e)}")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/camera/capture', methods=['POST'])
def capture_frame():
    global camera_active, last_frame

    if not camera_active or last_frame is None:
        return jsonify({
            'success': False,
            'message': 'Camera not active'
        }), 400

    try:
        with camera_lock:
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                }), 500

        # Save frame with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'capture_{timestamp}.jpg'
        cv2.imwrite(filename, frame)

        print(f"✓ Frame captured: {filename}")

        return jsonify({
            'success': True,
            'message': 'Frame captured',
            'filename': filename
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Capture error: {str(e)}'
        }), 500

@app.route('/api/detection/enable', methods=['POST'])
def enable_detection():
    global detection_enabled

    data = request.json
    detection_enabled = data.get('enabled', False)

    if detection_enabled:
        print("✓ Automatic blade detection enabled")
    else:
        print("✓ Automatic blade detection disabled")

    return jsonify({
        'success': True,
        'enabled': detection_enabled,
        'message': f"Detection {'enabled' if detection_enabled else 'disabled'}"
    })

@app.route('/api/detection/analyze', methods=['POST'])
def analyze_current_frame():
    global camera_active, last_frame, last_detection_result

    if not camera_active or last_frame is None:
        return jsonify({
            'success': False,
            'message': 'Camera not active or no frame available'
        }), 400

    try:
        with camera_lock:
            if last_frame is not None:
                frame_to_process = last_frame.copy()
            else:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                }), 500

        print(f"🔍 Analyzing frame: {frame_to_process.shape}")

        # Run detection
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        result = analyzer.analyze_frame()

        if result:
            last_detection_result = result
            print(f"✓ Detection: Valley between teeth {result.get('valley_id', 'N/A')}")
            print(f"  Coordinates: X={result['x_mm']:.2f}mm, Y={result['y_mm']:.2f}mm")
            print(f"  Distance: {result.get('distance_mm', 0):.2f}mm")
            print(f"  Total teeth: {result['num_teeth']}")
            print(f"  Grinder tip: ({result['grinder_tip_x_px']}, {result['grinder_tip_y_px']})")

            return jsonify({
                'success': True,
                'detection': result,
                'message': f"Detected valley between teeth {result.get('valley_id', 'N/A')}"
            })
        else:
            print("⚠ No teeth detected in frame")
            return jsonify({
                'success': False,
                'message': 'No teeth detected in frame'
            }), 404

    except Exception as e:
        print(f"✗ Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Detection error: {str(e)}'
        }), 500

@app.route('/api/detection/status', methods=['GET'])
def get_detection_status():
    global detection_enabled, last_detection_result

    return jsonify({
        'enabled': detection_enabled,
        'last_result': last_detection_result,
        'pixels_per_mm': pixels_per_mm
    })

@app.route('/api/detection/calibrate', methods=['POST'])
def calibrate_detection():
    global pixels_per_mm

    data = request.json
    new_ppm = data.get('pixels_per_mm')

    if new_ppm and new_ppm > 0:
        pixels_per_mm = float(new_ppm)
        print(f"✓ Calibration updated: {pixels_per_mm} pixels/mm")

        return jsonify({
            'success': True,
            'pixels_per_mm': pixels_per_mm,
            'message': 'Calibration updated'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid calibration value'
        }), 400

@app.route('/api/detection/update_grinder', methods=['POST'])
def update_grinder_position():
    """Force detection and update of grinder position"""
    global camera_active, last_frame, stored_grinder_tip

    if not camera_active or last_frame is None:
        return jsonify({
            'success': False,
            'message': 'Camera not active or no frame available'
        }), 400

    try:
        with camera_lock:
            if last_frame is not None:
                frame_to_process = last_frame.copy()
            else:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                }), 500

        print("🔄 Updating grinder position...")

        # Run detection WITHOUT using stored grinder
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        result = analyzer.analyze_frame(use_stored_grinder=False)

        if analyzer.grinder_tip:
            stored_grinder_tip = analyzer.grinder_tip
            save_grinder_position(stored_grinder_tip)

            return jsonify({
                'success': True,
                'grinder_tip': {
                    'x': int(stored_grinder_tip[0]),
                    'y': int(stored_grinder_tip[1])
                },
                'message': f'Grinder position updated: ({stored_grinder_tip[0]}, {stored_grinder_tip[1]})'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Could not detect grinder in frame'
            }), 404

    except Exception as e:
        print(f"✗ Grinder update error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Grinder update error: {str(e)}'
        }), 500

@app.route('/api/detection/grinder_status', methods=['GET'])
def get_grinder_status():
    """Get current stored grinder position"""
    global stored_grinder_tip

    if stored_grinder_tip:
        return jsonify({
            'success': True,
            'grinder_tip': {
                'x': int(stored_grinder_tip[0]),
                'y': int(stored_grinder_tip[1])
            },
            'stored': True
        })
    else:
        return jsonify({
            'success': True,
            'grinder_tip': None,
            'stored': False
        })

@app.route('/api/detection/send_auto', methods=['POST'])
def send_auto_detection():
    """Analyze frame and automatically send to robot"""
    global camera_active, last_frame, modbus_client

    if not camera_active or last_frame is None:
        return jsonify({
            'success': False,
            'message': 'Camera not active or no frame available'
        }), 400

    if not modbus_client or not modbus_client.connected:
        return jsonify({
            'success': False,
            'message': 'Not connected to robot'
        }), 400

    try:
        # Get frame
        with camera_lock:
            if last_frame is not None:
                frame_to_process = last_frame.copy()
            else:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                }), 500

        # Run detection
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        result = analyzer.analyze_frame()

        if not result:
            return jsonify({
                'success': False,
                'message': 'No teeth detected'
            }), 404

        # Send to robot
        modbus_result = modbus_client.write_detection(
            x_mm=result['x_mm'],
            y_mm=result['y_mm'],
            status=result['status']
        )

        if modbus_result and not modbus_result.isError():
            print(f"✓ Auto-detection sent: Valley {result.get('valley_id', 'N/A')} - X={result['x_mm']:.2f}mm, Y={result['y_mm']:.2f}mm")

            return jsonify({
                'success': True,
                'detection': result,
                'message': f"Valley sent to robot: {result['num_teeth']} teeth found"
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to send to robot'
            }), 500

    except Exception as e:
        print(f"✗ Auto-detection error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Auto-detection error: {str(e)}'
        }), 500

if __name__ == '__main__':
    import os

    print("=" * 80)
    print("🤖 BLADE GRINDER CONTROL SYSTEM - STANDALONE SERVER")
    print("=" * 80)
    print("\n✓ Starting Flask server...")
    print("✓ Dashboard will be available at: http://localhost:5000")
    print("✓ Also accessible at: http://0.0.0.0:5000")
    print("✓ API endpoints ready")
    print(f"\n✓ Working directory: {os.getcwd()}")
    print("✓ Looking for: robot_control_dashboard.html")
    print("\n" + "=" * 80)

    app.run(debug=True, host='0.0.0.0', port=5000)