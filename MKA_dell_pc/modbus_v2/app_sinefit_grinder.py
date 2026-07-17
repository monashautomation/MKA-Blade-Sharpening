# """
# Standalone Flask Backend for Robot Control Dashboard
# Blade grinder system – updated Modbus loop structure:

#   PC writes X/Y + status  →  START=1
#   Robot moves              →  sets GRINDER_READY=1
#   PC detects (camera on)   →  writes new X/Y, sends GRIND_START=1
#   Robot grinds, resets     →  GRIND_START=0  (ready for next cut)
#   Repeat per tooth

#   REG 141 = TEETH_INSPECT : Robot sets 1 to start calibration, 0 to stop.
#   REG 142 = ROBOT_ANGLE   : Robot continuously writes current blade angle (×10, signed).
#   While REG 141 = 1 the system records (angle, depth, pitch) only when
#   REG 142 changes by ≥ 0.1° — no duplicate samples at the same angle.
#   On falling edge (1→0) results are finalised; dashboard shows the
#   angle and pitch at the maximum-depth point.

# PERFORMANCE NOTES (Windows):
#   * The PySpin System + CameraList are now initialised ONCE and kept alive for
#     the whole process. Camera switches only DeInit/Init the camera itself —
#     no more 40 s Spinnaker re-discovery on every switch.
#   * debug_valid_mask.png write is disabled (DEBUG_SAVE_MASK flag).
#   * werkzeug per-request logging silenced, debug=False, threaded=True.
# """

# from flask import Flask, request, jsonify, Response
# from flask_cors import CORS
# from pymodbus.client import ModbusTcpClient
# import PySpin
# import cv2
# import numpy as np
# from scipy import ndimage
# from threading import Lock, Thread
# import time
# import atexit
# from dataclasses import dataclass
# from typing import List, Tuple

# from sine_fit_core import apply_sine_correction, draw_sine_overlay

# app = Flask(__name__)
# CORS(app)

# # ══════════════════════════════════════════════════════════════════════════════
# # ONE CONTROLLER, MANY VIEWERS
# # ------------------------------------------------------------------------------
# # Exactly one browser ("the controller") may perform actions that change the
# # machine (connect, config, start/stop, grind, camera, calibration, etc.).
# # Everyone else is a read-only VIEWER: they see the live camera + status but
# # their control actions are rejected with HTTP 423.
# #
# # EXCEPTION: EMERGENCY STOP (/api/estop) is allowed from ANY device, always.
# #
# # Enforcement is a single before_request hook keyed on the URL path, so no
# # individual route needs to change. The controller proves identity with an
# # opaque token (X-Control-Token header). A heartbeat keeps the lock alive;
# # if the controller's tab closes or freezes, the lock auto-frees after
# # CONTROL_TIMEOUT so the machine can't get stuck view-only forever.
# # ══════════════════════════════════════════════════════════════════════════════
# import uuid as _uuid
# import hmac as _hmac
# import os as _os
# control_lock  = Lock()
# control_state = {'token': None, 'name': None, 'last_heartbeat': 0.0}
# CONTROL_TIMEOUT = 15.0   # seconds without a heartbeat before control auto-releases

# # Control PIN — required to TAKE control (and to force a takeover). The current
# # holder refreshing its own lock does NOT need it. Set it by creating a file
# # named control_pin.txt next to this script; otherwise the default below is used.
# CONTROL_PIN = '2468'
# _pin_file = 'control_pin.txt'
# if _os.path.exists(_pin_file):
#     try:
#         _p = open(_pin_file).read().strip()
#         if _p:
#             CONTROL_PIN = _p
#             print(f"🔐 Control PIN loaded from {_pin_file}")
#     except Exception as _e:
#         print(f"⚠ Could not read {_pin_file}: {_e} — using default PIN")
# else:
#     print(f"🔐 Using DEFAULT control PIN. Create {_pin_file} with your own PIN to change it.")

# def _check_pin(pin):
#     # constant-time compare so the PIN can't be guessed by timing
#     return _hmac.compare_digest(str(pin), str(CONTROL_PIN))

# # POST paths that a viewer IS allowed to hit (everything else POST is gated)
# CONTROL_EXEMPT_PATHS = {
#     '/api/estop',              # EMERGENCY STOP — always allowed, any device
#     '/api/control/acquire',
#     '/api/control/release',
#     '/api/control/heartbeat',
#     '/api/control/status',
# }

# def _controller_active():
#     """True if someone currently holds control and their heartbeat is fresh."""
#     return (control_state['token'] is not None
#             and (time.time() - control_state['last_heartbeat']) < CONTROL_TIMEOUT)

# def _path_needs_control(path, method):
#     # Only mutating actions are gated. GET/HEAD/OPTIONS (status, camera frame,
#     # register reads, etc.) are always open so viewers can watch everything.
#     if method != 'POST':
#         return False
#     if not path.startswith('/api/'):
#         return False
#     if path in CONTROL_EXEMPT_PATHS:
#         return False
#     return True

# @app.before_request
# def _enforce_single_controller():
#     from flask import request
#     if not _path_needs_control(request.path, request.method):
#         return  # not a gated action → let it through
#     token = request.headers.get('X-Control-Token', '')
#     with control_lock:
#         if not _controller_active():
#             return jsonify({'success': False, 'control_error': True,
#                             'message': 'No device has taken control yet. '
#                                        'Click “Take Control” first.'}), 423
#         if token != control_state['token']:
#             return jsonify({'success': False, 'control_error': True,
#                             'message': f"View-only mode — {control_state['name']} "
#                                        f"currently has control."}), 423
#     # token matches active controller → allowed to proceed

# @app.route('/api/control/status', methods=['GET'])
# def control_status():
#     from flask import request
#     token = request.headers.get('X-Control-Token', '') or request.args.get('token', '')
#     with control_lock:
#         active = _controller_active()
#         return jsonify({
#             'success': True,
#             'has_controller': active,
#             'controller_name': control_state['name'] if active else None,
#             'you_are_controller': bool(active and token and token == control_state['token']),
#             'timeout_s': CONTROL_TIMEOUT,
#         })

# @app.route('/api/control/acquire', methods=['POST'])
# def control_acquire():
#     from flask import request
#     data  = request.json or {}
#     name  = (str(data.get('name') or 'Operator')).strip()[:40] or 'Operator'
#     force = bool(data.get('force', False))
#     pin   = data.get('pin', '')
#     hdr_token = request.headers.get('X-Control-Token', '')
#     with control_lock:
#         active = _controller_active()
#         # Current holder refreshing its own lock (heartbeat / rename) → no PIN.
#         if active and hdr_token and hdr_token == control_state['token']:
#             control_state['last_heartbeat'] = time.time()
#             if name:
#                 control_state['name'] = name
#             return jsonify({'success': True, 'token': hdr_token,
#                             'name': control_state['name'], 'reacquired': True})
#         # Any real take-over of control requires the PIN.
#         if not _check_pin(pin):
#             return jsonify({'success': False, 'pin_error': True,
#                             'message': 'Incorrect control PIN.'}), 403
#         if active and not force:
#             return jsonify({'success': False, 'control_error': True,
#                             'held_by': control_state['name'],
#                             'message': f"{control_state['name']} currently has control."}), 409
#         # PIN OK and (free, expired, or forced) → grant a fresh token.
#         new_token = _uuid.uuid4().hex
#         control_state['token'] = new_token
#         control_state['name']  = name
#         control_state['last_heartbeat'] = time.time()
#         print(f"🕹  Control acquired by '{name}'"
#               f"{' (FORCED takeover)' if (active and force) else ''}")
#         return jsonify({'success': True, 'token': new_token, 'name': name,
#                         'forced': bool(active and force)})

# @app.route('/api/control/heartbeat', methods=['POST'])
# def control_heartbeat():
#     from flask import request
#     token = request.headers.get('X-Control-Token', '') or (request.json or {}).get('token', '')
#     with control_lock:
#         if _controller_active() and token == control_state['token']:
#             control_state['last_heartbeat'] = time.time()
#             return jsonify({'success': True, 'you_are_controller': True})
#         active = _controller_active()
#         return jsonify({'success': True, 'you_are_controller': False,
#                         'has_controller': active,
#                         'controller_name': control_state['name'] if active else None})

# @app.route('/api/control/release', methods=['POST'])
# def control_release():
#     from flask import request
#     # Accept token via header, ?token= (for navigator.sendBeacon), or JSON body.
#     token = (request.headers.get('X-Control-Token', '')
#              or request.args.get('token', '')
#              or (request.get_json(silent=True) or {}).get('token', ''))
#     with control_lock:
#         if control_state['token'] is not None and token == control_state['token']:
#             print(f"🕹  Control released by '{control_state['name']}'")
#             control_state['token'] = None
#             control_state['name']  = None
#             control_state['last_heartbeat'] = 0.0
#             return jsonify({'success': True, 'released': True})
#         return jsonify({'success': True, 'released': False})

# # ── Debug flags ───────────────────────────────────────────────────────────────
# DEBUG_SAVE_MASK = False   # True → write debug_valid_mask.png on every analysis (SLOW on Windows)

# # ── Grinder detection tuning ──────────────────────────────────────────────────
# # Search area for the grinder's left edge. The scan starts to the RIGHT of the
# # blade's actual right edge (+ GRINDER_GAP_FRAC) and covers rows from
# # GRINDER_SCAN_TOP_FRAC*h down to GRINDER_SCAN_BOT_FRAC*h.
# GRINDER_SCAN_TOP_FRAC = 1.0 / 3.0   # top of scan band (fraction of image height)
# GRINDER_SCAN_BOT_FRAC = 1.0         # bottom of scan band
# GRINDER_GAP_FRAC      = 0.02        # gap right of the blade edge (fraction of width)
# GRINDER_MIN_DARK_PX   = 5           # min dark pixels in a row to count as grinder
# BLADE_LIMIT_FRAC      = 0.45        # blade lives in the left N of the frame

# # Threshold for "grinder dark". 'otsu' self-calibrates on the scan region and is
# # strongly recommended: the grinder is near-black (~18 grey) while the background
# # is ~150-190 WITH a top-to-bottom gradient, so any fixed fraction of a
# # background mean sampled at the top can fall below the background at the bottom
# # and make the scan return vignette contours instead of the grinder.
# # Set to a number (e.g. 100) to force an absolute grey level instead.
# GRINDER_THRESH_MODE = 'otsu'        # 'otsu' | <int absolute grey level>

# # A real V-grinder has a known included angle. If the fitted V is outside this
# # range the fit is not a grinder (glare, blade, empty frame) → report no tip
# # rather than silently handing the robot a bogus coordinate.
# GRINDER_ANGLE_MIN = 75.0
# GRINDER_ANGLE_MAX = 105.0
# # If the V-fit fails, fall back to the leftmost dark pixel (the ROUNDED NOSE,
# # ~12 px / 0.16 mm right of the true vertex)? False = safer: report no tip.
# GRINDER_ALLOW_NOSE_FALLBACK = False

# # ── Global state ──────────────────────────────────────────────────────────────
# modbus_client    = None
# client_connected = False

# pyspin_system   = None   # kept alive for the whole process (see _get_pyspin_system)
# pyspin_cam      = None
# pyspin_cam_list = None   # kept alive for the whole process
# camera_lock     = Lock()
# camera_active   = False
# camera_thread   = None
# last_frame      = None
# camera_config   = {'frame_rate': 30.0 }

# blade_analyzer        = None
# last_detection_result = None
# last_sine_overlay     = None   # {'apexes','valleys','info'} from last successful sine fit (for live overlay)
# last_overlay_pts      = None   # NoSine build: {'apexes','valleys'} deduped tops + candidate valleys for overlay
# sine_overlay_enabled  = False  # toggled by the dashboard overlay selector ('sine' mode); hides the live envelope otherwise
# live_sine_overlay     = None   # {'apexes','valleys','info','t'} fresh per-frame fit cached by /teeth_profiles for the live feed
# detection_enabled     = False
# pixels_per_mm           = 76.812
# pitch_mm                = 0.0    # 0.0 = disabled; set from dashboard
# grinder_positions_file  = 'grinder_positions.json'   # was grinder_position.json (singular)
# stored_grinder_tips     = {}                         # {grinder_id: (x, y)}

# # Camera-per-grinder mapping. Edit camera_serials.json after first run to assign
# # FLIR serial numbers to each grinder. Empty string = fall back to index (gid-1).
# camera_serials_file = 'camera_serials.json'
# CAMERA_SERIALS      = {1: '', 2: ''}
# current_grinder_id  = 1

# # ── Inspection state ──────────────────────────────────────────────────────────
# inspection_active       = False
# inspection_thread       = None
# last_inspection_summary = None
# last_teeth_inspect_reg  = False
# last_robot_angle        = 0.0

# # ── Post-cut depth validation ─────────────────────────────────────────────────
# last_grind_depth_mm     = None   # pre-cut groove depth (depth sent with last GRIND_START)
# last_commanded_depth_mm = None   # commanded depth from last config (REG 131)
# last_post_cut_result    = None   # last post-cut measurement (for dashboard + REG 150)




# # ── Grinder position persistence ──────────────────────────────────────────────
# def load_grinder_positions():
#     """Load per-grinder tip positions. Migrates old single-file format if found."""
#     global stored_grinder_tips
#     import os, json
#     if os.path.exists(grinder_positions_file):
#         try:
#             with open(grinder_positions_file, 'r') as f:
#                 data = json.load(f)
#                 stored_grinder_tips = {int(k): tuple(v) for k, v in data.items()}
#                 print(f"✓ Loaded grinder positions: {stored_grinder_tips}")
#         except Exception as e:
#             print(f"⚠ Could not load grinder positions: {e}")
#             stored_grinder_tips = {}
#     elif os.path.exists('grinder_position.json'):
#         # Migrate old single-position file → grinder 1
#         try:
#             with open('grinder_position.json', 'r') as f:
#                 data = json.load(f)
#                 stored_grinder_tips = {1: tuple(data['grinder_tip'])}
#                 print(f"✓ Migrated old grinder position → grinder 1: {stored_grinder_tips[1]}")
#                 save_grinder_positions()
#         except Exception as e:
#             print(f"⚠ Migration failed: {e}")
#             stored_grinder_tips = {}
#     else:
#         stored_grinder_tips = {}


# def save_grinder_positions():
#     import json
#     try:
#         data = {str(k): [int(v[0]), int(v[1])] for k, v in stored_grinder_tips.items()}
#         with open(grinder_positions_file, 'w') as f:
#             json.dump(data, f, indent=4)
#         print(f"✓ Saved grinder positions: {stored_grinder_tips}")
#     except Exception as e:
#         print(f"⚠ Could not save grinder positions: {e}")


# def get_current_grinder_tip():
#     return stored_grinder_tips.get(current_grinder_id)


# def set_current_grinder_tip(tip):
#     stored_grinder_tips[current_grinder_id] = (int(tip[0]), int(tip[1]))
#     save_grinder_positions()


# def load_camera_serials():
#     global CAMERA_SERIALS
#     import os, json
#     if os.path.exists(camera_serials_file):
#         try:
#             with open(camera_serials_file, 'r') as f:
#                 data = json.load(f)
#                 CAMERA_SERIALS = {int(k): str(v) for k, v in data.items()}
#                 print(f"✓ Loaded camera serials: {CAMERA_SERIALS}")
#         except Exception as e:
#             print(f"⚠ Could not load camera serials: {e}")
#     else:
#         save_camera_serials()  # write default file so user can edit it


# def save_camera_serials():
#     import json
#     try:
#         with open(camera_serials_file, 'w') as f:
#             json.dump({str(k): v for k, v in CAMERA_SERIALS.items()}, f, indent=4)
#     except Exception as e:
#         print(f"⚠ Could not save camera serials: {e}")


# load_grinder_positions()
# load_camera_serials()


# def _dedupe_clusters(profiles, y_threshold_mm=0.2, grinder_tip=None, select=None):
#     """
#     Collapse groups of teeth at nearly-identical Y positions into a single
#     representative tooth per cluster. This removes duplicate detections
#     without skipping any real teeth.

#     Clustering rule: consecutive teeth within `y_threshold_mm` of each other
#     are treated as the same physical tooth.

#     Representative selection:
#       - If grinder_tip provided → closest to grinder
#       - Otherwise → rightmost (max X)  [original behavior]

#     Args:
#         profiles: list of ToothProfile, ordered by Y
#         y_threshold_mm: cluster spacing threshold in millimeters
#         grinder_tip: optional (x, y) to bias selection toward grinder

#     Returns:
#         New list with one representative tooth per cluster.
#     """
#     global pixels_per_mm

#     if not profiles:
#         return []

#     threshold_px = y_threshold_mm * pixels_per_mm
#     gt = grinder_tip if grinder_tip else get_current_grinder_tip()

#     deduped = []
#     i = 0
#     while i < len(profiles):
#         # Build cluster starting at i
#         cluster = [profiles[i]]
#         j = i + 1
#         while j < len(profiles):
#             if abs(cluster[-1].grinding_point[1] - profiles[j].grinding_point[1]) < threshold_px:
#                 cluster.append(profiles[j])
#                 j += 1
#             else:
#                 break

#         # Pick representative
#         if len(cluster) == 1:
#             best = cluster[0]
#         elif select == 'tallest':
#             # Most-protruding tip = the real tooth; a duplicate is a smaller bump beside it
#             best = max(cluster, key=lambda t: t.height)
#         elif gt:
#             # Closest to grinder
#             best = min(cluster, key=lambda t: (
#                                                       (t.grinding_point[0] - gt[0]) ** 2 +
#                                                       (t.grinding_point[1] - gt[1]) ** 2
#                                               ) ** 0.5)
#         else:
#             # Rightmost (original behavior)
#             best = max(cluster, key=lambda t: t.grinding_point[0])

#         deduped.append(best)
#         i = j  # jump past the entire cluster

#     return deduped


# # ── Blade detection ───────────────────────────────────────────────────────────
# @dataclass
# class ToothProfile:
#     tooth_id:        int
#     apex_point:      Tuple[int, int]
#     top_valley:      Tuple[int, int]
#     bottom_valley:   Tuple[int, int]
#     angle:           float
#     grinding_point:  Tuple[int, int]
#     height:          float
#     move_to_grinder: Tuple[float, float]


# class SerratedBladeAnalyzer:
#     def __init__(self, image):
#         self.image  = image
#         self.gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         self.height, self.width = self.gray.shape
#         self.teeth_profiles     = []
#         self.grinder_tip        = None
#         self.blade_edge_points  = None
#         self.grinder_edge_points = None
#         self.grinder_upper_line = None
#         self.grinder_lower_line = None
#         self.grinder_edge_center = None
#         self.grinder_roi = None
#         self.grinder_thresh = None
#         self.grad = 0.0
#         self.m1 = 0
#         self.m2 = 0

#     def preprocess_image(self, blur_kernel=5):
#         """
#         Preprocess for grey-background images.

#         Uses Otsu thresholding (auto-selects the blade/background boundary)
#         instead of adaptive threshold which required near-white background.
#         Morphological close removes small burr noise before edge extraction.
#         """
#         self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)

#         # Otsu: automatically finds threshold between dark blade and grey background
#         self._otsu_thresh, self.binary = cv2.threshold(
#             self.blurred, 0, 255,
#             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#         )

#         # Morphological close: fills in burr gaps, smooths edge noise
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel, iterations=2)

#         return self.binary

#     def detect_blade_and_grinder(self, sampling_step=1):
#         """
#         Detect blade edge (left) and grinder tip (right) for grey-background images.

#         Blade edge
#         ----------
#         Scans every row for the rightmost dark (blade) pixel in the left
#         BLADE_LIMIT_FRAC of the image. Uses the Otsu binary from preprocess_image().

#         Grinder edge
#         ------------
#         The grinder is a separate, near-black object to the right of the blade.
#         For each row in the scan band we take the LEFTMOST dark pixel — that is
#         the grinder's left edge. Those points feed _fit_grinder_v, which fits a
#         line to each flank of the V and intersects them; the intersection (the
#         virtual sharp vertex) is the tip, NOT the leftmost pixel.

#         Thresholding (see GRINDER_THRESH_MODE)
#         --------------------------------------
#         Otsu is computed on the scan region itself. The previous approach —
#         0.82 x (mean of a background patch sampled at the TOP of the frame) —
#         is unsafe: measured on a real frame the background runs ~191 grey at the
#         top but ~157 at the bottom (vignetting), so the threshold (157) landed
#         BELOW the background it was applied to and every "grinder" point was
#         actually background. Otsu on the region self-calibrates and sits in the
#         wide gap between grinder (~18) and background (~150+).
#         """
#         h, w = self.gray.shape
#         blade_limit = int(w * BLADE_LIMIT_FRAC)

#         # ── Blade edge ────────────────────────────────────────────────────────────
#         blade_edge = []
#         for y in range(0, h, sampling_step):
#             row = self.binary[y, :blade_limit]
#             dark = np.where(row > 0)[0]
#             if len(dark) > 10:  # skip rows with barely any blade
#                 blade_edge.append((int(dark.max()), y))

#         self.blade_edge_points = np.array(blade_edge) if blade_edge else None

#         # ── Grinder scan area ─────────────────────────────────────────────────────
#         # Anchor the left boundary to where the blade REALLY ends in this frame
#         # (+ a gap), so the scan follows the grinder wherever it advances while
#         # keeping the blade teeth out of it.
#         if self.blade_edge_points is not None and len(self.blade_edge_points):
#             blade_right = int(self.blade_edge_points[:, 0].max())
#         else:
#             blade_right = blade_limit
#         gap_px  = max(8, int(w * GRINDER_GAP_FRAC))
#         scan_x0 = min(blade_right + gap_px, w - 1)
#         scan_y0 = max(0, int(h * GRINDER_SCAN_TOP_FRAC))
#         scan_y1 = min(h, int(h * GRINDER_SCAN_BOT_FRAC))
#         self.grinder_roi = (scan_x0, scan_y0, w - 1, scan_y1)   # for overlay/debug

#         # ── Grinder threshold ─────────────────────────────────────────────────────
#         region = self.gray[scan_y0:scan_y1, scan_x0:]
#         if GRINDER_THRESH_MODE == 'otsu' and region.size:
#             _t, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             grinder_thresh = int(_t)
#         elif isinstance(GRINDER_THRESH_MODE, (int, float)):
#             grinder_thresh = int(GRINDER_THRESH_MODE)
#         else:
#             grinder_thresh = 100
#         self.grinder_thresh = grinder_thresh

#         # ── Grinder edge (leftmost dark pixel per row) ────────────────────────────
#         grinder_points = []
#         for y in range(scan_y0, scan_y1, sampling_step):
#             row = self.gray[y, scan_x0:]
#             dark = np.where(row < grinder_thresh)[0]
#             if len(dark) > GRINDER_MIN_DARK_PX:
#                 grinder_points.append((int(dark.min()) + scan_x0, y))

#         self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

#         # ── V-fit: tip = intersection of the two fitted flank lines ───────────────
#         self.grinder_tip = None
#         self.grinder_edge_center = None
#         self.grinder_upper_line = None
#         self.grinder_lower_line = None

#         if self.grinder_edge_points is not None and len(self.grinder_edge_points) >= 6:
#             tip, upper, lower, self.grad, self.m1, self.m2 = \
#                 self._fit_grinder_v(self.grinder_edge_points)

#             # Reject an implausible V (glare / blade / empty frame). Otsu always
#             # returns *some* threshold, so this is what stops a bogus tip from
#             # reaching the robot.
#             if tip is not None and not (GRINDER_ANGLE_MIN <= self.grad <= GRINDER_ANGLE_MAX):
#                 print(f"⚠ Grinder V rejected: included angle {self.grad:.1f}° outside "
#                       f"{GRINDER_ANGLE_MIN}-{GRINDER_ANGLE_MAX}°")
#                 tip = None

#             if tip is not None:
#                 self.grinder_tip = tip                 # the virtual sharp vertex
#                 self.grinder_edge_center = tip
#                 self.grinder_upper_line = upper
#                 self.grinder_lower_line = lower
#             elif GRINDER_ALLOW_NOSE_FALLBACK:
#                 # Rounded nose — ~12 px (0.16 mm) right of the true vertex. Off by
#                 # default: a wrong tip is worse than no tip on a live machine.
#                 min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
#                 self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
#                 min_x = self.grinder_edge_points[min_x_idx, 0]
#                 tip_pts = self.grinder_edge_points[
#                     np.abs(self.grinder_edge_points[:, 0] - min_x) < 10]
#                 self.grinder_edge_center = (
#                     int(np.mean(tip_pts[:, 0])), int(np.mean(tip_pts[:, 1])))
#                 print("⚠ Grinder V-fit failed — using leftmost-pixel fallback (less accurate)")

#         return self.blade_edge_points, self.grinder_tip

#     def _generate_coordinates_sine(self, grinder_tip):
#         global pixels_per_mm, pitch_mm

#         gt_y = grinder_tip[1]
#         gt_x = grinder_tip[0]

#         corrected_apexes, corrected_valleys, info, above_profiles = apply_sine_correction(
#             self.teeth_profiles,
#             grinder_tip,
#             pixels_per_mm,
#             pitch_mm,
#             blade_edge_points=self.blade_edge_points,
#             min_teeth_for_fit=2,
#             baseline_poly_degree=1,
#         )

#         if corrected_apexes is None or corrected_valleys is None or len(corrected_apexes) < 2:
#             return None

#         self._sine_corrected_apexes = corrected_apexes
#         self._sine_corrected_valleys = corrected_valleys
#         self._sine_info = info

#         global last_sine_overlay
#         last_sine_overlay = {'apexes': corrected_apexes,
#                              'valleys': corrected_valleys, 'info': info}

#         sorted_valleys = sorted(corrected_valleys, key=lambda v: v[1])
#         closest_valley = None
#         min_distance = float('inf')

#         for (vx, vy) in sorted_valleys:
#             move_x_mm = (gt_x - vx) / pixels_per_mm
#             move_y_mm = (gt_y - vy) / pixels_per_mm
#             if move_y_mm < 0.5:
#                 continue
#             dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
#             if dist < min_distance:
#                 min_distance = dist
#                 depth_x_mm = abs(info['A_mean']) / pixels_per_mm
#                 apexes_above = [a for a in corrected_apexes if a[1] < vy]
#                 ct_lbl = len(apexes_above)
#                 closest_valley = {
#                     'valley_x': vx,
#                     'valley_y': vy,
#                     'move_x_mm': move_x_mm,
#                     'move_y_mm': move_y_mm,
#                     'depth_x_mm': depth_x_mm,
#                     'between_teeth': f"{ct_lbl}-{ct_lbl + 1}",
#                     'distance_mm': dist,
#                 }

#         if not closest_valley:
#             return None

#         num_teeth = len([a for a in corrected_apexes if a[1] < gt_y])

#         return {
#             'valley_id': closest_valley['between_teeth'],
#             'x_mm': round(float(closest_valley['move_y_mm']), 2),
#             'y_mm': round(float(closest_valley['move_x_mm']), 2),
#             'depth_x_mm': round(float(closest_valley['depth_x_mm']), 2),
#             'valley_x_px': int(closest_valley['valley_x']),
#             'valley_y_px': int(closest_valley['valley_y']),
#             'grinder_tip_x_px': int(grinder_tip[0]),
#             'grinder_tip_y_px': int(grinder_tip[1]),
#             'num_teeth': num_teeth,
#             'distance_mm': round(float(closest_valley['distance_mm']), 2),
#             'status': 1,
#             'all_valleys': [],
#             'sine_fit': True,
#             'pitch_mm': pitch_mm,
#         }
#     def _fit_grinder_v(self, points, min_points_per_side=3, max_iter=3, trim_frac=0.15):
#         """
#         Fit two lines (upper + lower halves of V) to grinder edge points and return
#         the intersection (V-apex = grinder tip).

#         Iterative refinement:
#           1. Find point of minimum X → rough apex estimate
#           2. Split points into above/below apex by Y
#           3. Fit a line to each half using least squares on x = m*y + b
#              (swapped because the V is near-vertical; avoids infinite slopes)
#           4. Trim outliers (worst `trim_frac` residuals) and refit
#           5. Solve intersection of the two lines for refined apex
#           6. Repeat using refined apex as new split point

#         Args:
#             points: Nx2 array of (x, y) grinder edge points
#             min_points_per_side: need at least this many points per half
#             max_iter: refinement iterations (2–3 is enough)
#             trim_frac: fraction of worst-fit points to discard per iteration

#         Returns:
#             (tip, upper_line_params, lower_line_params) or (None, None, None) on failure
#             - tip = (x, y) integer tuple (V-apex)
#             - *_line_params = (m, b) for x = m*y + b
#         """
#         if len(points) < 2 * min_points_per_side:
#             return None, None, None, 0.0, 0.0, 0.0

#         pts = np.asarray(points, dtype=float)
#         xs, ys = pts[:, 0], pts[:, 1]

#         # ── Initial split: use argmin(x) as rough apex ────────────────────────────
#         split_y = ys[np.argmin(xs)]

#         upper_line = None
#         lower_line = None
#         tip = None

#         for _ in range(max_iter):
#             upper_mask = ys < split_y
#             lower_mask = ys > split_y

#             if upper_mask.sum() < min_points_per_side or lower_mask.sum() < min_points_per_side:
#                 break

#             upper_line = self._fit_line_trimmed(xs[upper_mask], ys[upper_mask], trim_frac)
#             lower_line = self._fit_line_trimmed(xs[lower_mask], ys[lower_mask], trim_frac)

#             if upper_line is None or lower_line is None:
#                 break

#             # Solve intersection: x = m1*y + b1 = m2*y + b2
#             # → y = (b1 - b2) / (m2 - m1), then x = m1*y + b1
#             m1, b1 = upper_line
#             m2, b2 = lower_line
#             denom = (m2 - m1)
#             if abs(denom) < 1e-6:
#                 # Lines are parallel — can't find intersection
#                 break

#             y_apex = (b1 - b2) / denom
#             x_apex = m1 * y_apex + b1
#             tip = (int(round(x_apex)), int(round(y_apex)))

#             # Use new apex as split point for next iteration
#             new_split = y_apex
#             if abs(new_split - split_y) < 0.5:
#                 break  # converged
#             split_y = new_split

#         if tip is None or upper_line is None or lower_line is None:
#             return None, None, None, 0.0, 0.0, 0.0

#         # Sanity check: the fitted tip should lie near the observed points
#         # (within the y-range and not wildly off in x)
#         y_min, y_max = ys.min(), ys.max()
#         if not (y_min - 20 <= tip[1] <= y_max + 20):
#             return None, None, None, 0.0, 0.0, 0.0

#         x_min = xs.min()
#         if tip[0] < x_min - 30 or tip[0] > xs.max() + 30:
#             return None, None, None, 0.0, 0.0, 0.0

#         import math
#         gradient = math.degrees(math.atan(abs((m1-m2)/(1+m1*m2))))
#         return tip, upper_line, lower_line, gradient, m1 , m2

#     @staticmethod
#     def _fit_line_trimmed(xs, ys, trim_frac=0.15):
#         """
#         Fit x = m*y + b using least squares, then discard the worst `trim_frac`
#         of points by residual and refit once. Returns (m, b) or None.

#         We fit x as a function of y (not y = f(x)) because the V-shape edges
#         are near-vertical in the image; x = m*y + b handles vertical lines
#         gracefully while y = m*x + b would blow up.
#         """
#         xs = np.asarray(xs, dtype=float)
#         ys = np.asarray(ys, dtype=float)

#         if len(xs) < 2:
#             return None

#         # First pass
#         try:
#             m, b = np.polyfit(ys, xs, 1)
#         except (np.linalg.LinAlgError, ValueError):
#             return None

#         if len(xs) < 4:
#             return (float(m), float(b))

#         # Trim worst residuals and refit
#         residuals = np.abs(xs - (m * ys + b))
#         threshold = np.quantile(residuals, 1.0 - trim_frac)
#         keep = residuals <= threshold

#         if keep.sum() < 2:
#             return (float(m), float(b))

#         try:
#             m2, b2 = np.polyfit(ys[keep], xs[keep], 1)
#             return (float(m2), float(b2))
#         except (np.linalg.LinAlgError, ValueError):
#             return (float(m), float(b))

#     def extract_tooth_profiles(self, window_size=10):
#         if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
#             return []
#         x_coords = self.blade_edge_points[:, 0]
#         y_coords = self.blade_edge_points[:, 1]
#         x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

#         valid_mask = np.ones(len(y_coords), dtype=bool)
#         if self.image is not None:
#             img_array = np.array(self.image) if not isinstance(self.image, np.ndarray) else self.image
#             gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if img_array.ndim == 3 else img_array
#             for i, y in enumerate(y_coords.astype(int)):
#                 if 0 <= y < gray.shape[0] and np.all(gray[y, :] == 255):
#                     valid_mask[i] = False

#         n = len(x_smooth)
#         baseline_half = min(max(window_size * 4, n // 6), n // 3)
#         pad    = baseline_half
#         padded = np.pad(x_smooth, pad, mode='reflect')
#         baseline = np.array([np.median(padded[max(0, i): i + 2 * pad + 1]) for i in range(n)])
#         if not valid_mask.all():
#             valid_idx = np.where(valid_mask)[0]
#             if len(valid_idx) > 0:
#                 for i in np.where(~valid_mask)[0]:
#                     baseline[i] = baseline[valid_idx[np.argmin(np.abs(valid_idx - i))]]

#         deviation  = x_smooth - baseline
#         dev_valid  = deviation[valid_mask] if valid_mask.any() else deviation
#         prominence = max(2.0, (np.percentile(dev_valid, 75) - np.percentile(dev_valid, 25)) * 0.20) if len(dev_valid) > 4 else 2.0

#         peaks, valleys = [], []
#         for i in range(window_size, n - window_size):
#             if not valid_mask[i]: continue
#             window_dev = deviation[i - window_size: i + window_size]
#             if deviation[i] == np.max(window_dev) and deviation[i] > prominence:
#                 peaks.append(i)
#             elif deviation[i] == np.min(window_dev) and deviation[i] < -prominence:
#                 valleys.append(i)
#         peaks   = self._filter_close_points(peaks,   window_size)
#         valleys = self._filter_close_points(valleys, window_size)

#         # Debug visualisation — DISABLED by default. Writing a PNG on every
#         # analysis call (overlay poll = every 300 ms) is a major slowdown,
#         # especially on Windows where Defender scans each file write.
#         if DEBUG_SAVE_MASK and self.image is not None:
#             mask_img = img_array.copy()
#             for i, y in enumerate(y_coords.astype(int)):
#                 if 0 <= y < mask_img.shape[0]:
#                     x = int(x_coords[i])
#                     if 0 <= x < mask_img.shape[1]:
#                         cv2.circle(mask_img, (x, y), 3, (0,255,0) if valid_mask[i] else (0,0,255), -1)
#             for i, y in enumerate(y_coords.astype(int)):
#                 bx = int(baseline[i])
#                 if 0 <= y < mask_img.shape[0] and 0 <= bx < mask_img.shape[1]:
#                     cv2.circle(mask_img, (bx, y), 2, (0,165,255), -1)
#             cv2.imwrite("debug_valid_mask.png", mask_img)

#         tooth_profiles = []
#         for tooth_id, peak_idx in enumerate(peaks, start=1):
#             valleys_above = [v for v in valleys if v < peak_idx]
#             valleys_below = [v for v in valleys if v > peak_idx]
#             if len(valleys_above) == 0 and len(valleys_below) > 0:
#                 s  = [x_smooth[idx] for idx in range(0, min(window_size//2, peak_idx))]
#                 sy = [y_coords[idx] for idx in range(0, min(window_size//2, peak_idx))]
#                 top_valley    = (int(np.mean(s)),int(np.mean(sy))) if s else (int(x_smooth[0]),int(y_coords[0]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]),int(y_coords[valleys_below[0]]))
#             elif len(valleys_below) == 0 and len(valleys_above) > 0:
#                 start = peak_idx; end = min(len(x_smooth)-1, peak_idx+window_size*2)
#                 s  = [x_smooth[idx] for idx in range(start,end)]
#                 sy = [y_coords[idx] for idx in range(start,end)]
#                 top_valley    = (int(x_smooth[valleys_above[-1]]),int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(np.mean(s)),int(np.mean(sy))) if s else (int(x_smooth[end]),int(y_coords[end]))
#             elif len(valleys_above) > 0 and len(valleys_below) > 0:
#                 top_valley    = (int(x_smooth[valleys_above[-1]]),int(y_coords[valleys_above[-1]]))
#                 bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
#             else:
#                 continue
#             tooth_point = (int(x_smooth[peak_idx]),int(y_coords[peak_idx]))
#             height      = abs(tooth_point[0] - ((top_valley[0]+bottom_valley[0])/2))
#             angle       = self._calculate_tooth_angle(top_valley,tooth_point,bottom_valley)
#             move_to_grinder = ((self.grinder_tip[0]-tooth_point[0],self.grinder_tip[1]-tooth_point[1]) if self.grinder_tip else (0,0))
#             tooth_profiles.append(ToothProfile(
#                 tooth_id=tooth_id,apex_point=tooth_point,
#                 top_valley=top_valley,bottom_valley=bottom_valley,
#                 angle=angle,grinding_point=tooth_point,
#                 height=height,move_to_grinder=move_to_grinder))
#         return tooth_profiles

#     def draw_grinder_v_overlay(self, img):
#         """Draw the fitted V (two lines + apex point) on a BGR image in-place."""
#         if self.grinder_upper_line is None or self.grinder_lower_line is None:
#             return img
#         if self.grinder_edge_points is None or len(self.grinder_edge_points) == 0:
#             return img

#         h, w = img.shape[:2]
#         ys = self.grinder_edge_points[:, 1]
#         y_min, y_max = int(ys.min()), int(ys.max())

#         # Upper line (x = m*y + b) — draw in cyan
#         m1, b1 = self.grinder_upper_line
#         if self.grinder_tip:
#             y_end = self.grinder_tip[1]
#         else:
#             y_end = (y_min + y_max) // 2
#         pt1 = (int(round(m1 * y_min + b1)), y_min)
#         pt2 = (int(round(m1 * y_end + b1)), int(y_end))
#         cv2.line(img, pt1, pt2, (255, 255, 0), 2)

#         # Lower line — draw in magenta
#         m2, b2 = self.grinder_lower_line
#         if self.grinder_tip:
#             y_start = self.grinder_tip[1]
#         else:
#             y_start = (y_min + y_max) // 2
#         pt3 = (int(round(m2 * y_start + b2)), int(y_start))
#         pt4 = (int(round(m2 * y_max + b2)), y_max)
#         cv2.line(img, pt3, pt4, (255, 0, 255), 2)

#         # Apex (V-tip) — big yellow circle with crosshair
#         if self.grinder_tip:
#             tx, ty = int(self.grinder_tip[0]), int(self.grinder_tip[1])
#             cv2.circle(img, (tx, ty), 10, (0, 255, 255), 2)
#             cv2.line(img, (tx - 15, ty), (tx + 15, ty), (0, 255, 255), 1)
#             cv2.line(img, (tx, ty - 15), (tx, ty + 15), (0, 255, 255), 1)
#             cv2.putText(img, "V-APEX ", (tx + 12, ty - 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#             cv2.putText(img, f"{float(self.grad)}", (tx + 100, ty - 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#         # Raw edge points — small green dots
#         for x, y in self.grinder_edge_points:
#             cv2.circle(img, (int(x), int(y)), 1, (0, 200, 0), -1)

#         return img

#     def _filter_close_points(self, points, min_distance):
#         if not points: return []
#         filtered = [points[0]]
#         for p in points[1:]:
#             if p - filtered[-1] >= min_distance: filtered.append(p)
#         return filtered

#     def _calculate_tooth_angle(self, top_valley, tooth_point, bottom_valley):
#         try:
#             v1 = np.array(tooth_point) - np.array(top_valley)
#             v2 = np.array(bottom_valley) - np.array(tooth_point)
#             return float(np.degrees(np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])))
#         except: return 0.0



#     def _groove_sharpness_score(self, ct_tip, nt_tip, perp_half=16):
#         """
#         Measure edge contrast in the V-groove by sampling perpendicular to
#         the blade edge at every detected edge pixel within the groove y-range.

#         At each edge point (x, y):
#           dark side  = gray[y, x - perp_half]  (a few px into the blade)
#           bright side = gray[y, x + perp_half]  (a few px into the background)
#           contrast   = bright - dark

#         Mean contrast = sharpness score.  Higher = sharper.
#         Immune to burrs because it measures intensity difference across the
#         edge, not the geometry of the edge itself.

#         perp_half : pixels to step each side of the edge (default 8).
#                     Increase to average over a wider sampling band.
#         """
#         if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
#             return None

#         h, w = self.gray.shape

#         # Filter edge points to the groove y-range (between the two tooth tips)
#         y_min = min(ct_tip[1], nt_tip[1])
#         y_max = max(ct_tip[1], nt_tip[1])
#         ys   = self.blade_edge_points[:, 1]
#         mask = (ys >= y_min) & (ys <= y_max)
#         groove_pts = self.blade_edge_points[mask]

#         if len(groove_pts) == 0:
#             return None

#         contrasts = []
#         for x, y in groove_pts:
#             xi, yi = int(round(x)), int(round(y))
#             if not (0 <= yi < h):
#                 continue
#             x_dark   = max(0,     xi - perp_half)  # into blade (dark side)
#             x_bright = min(w - 1, xi + perp_half)  # into background (bright side)
#             contrast = float(self.gray[yi, x_bright]) - float(self.gray[yi, x_dark])
#             if contrast > 0:   # guard against inverted or invalid samples
#                 contrasts.append(contrast)

#         return float(np.mean(contrasts)) if contrasts else None

#     def _get_closest_pair_stats(self, sharp_threshold=150.0):
#         """
#         Uses cluster-deduped tooth list, then iterates all consecutive pairs.
#         Returns pitch, depth, sharpness for the pair closest to grinder.
#         """
#         global pixels_per_mm

#         grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
#         if not grinder_tip or len(self.teeth_profiles) < 2:
#             return None

#         # Dedupe clusters first
#         deduped = _dedupe_clusters(
#             self.teeth_profiles,
#             y_threshold_mm=1,
#             grinder_tip=grinder_tip,
#         )

#         if len(deduped) < 2:
#             return None

#         closest = None
#         min_dist = float("inf")

#         for i in range(len(deduped) - 1):
#             ct = deduped[i]
#             nt = deduped[i + 1]

#             valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
#             valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
#             move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
#             move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
#             dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5

#             if dist < min_dist:
#                 min_dist = dist
#                 closest = {
#                     "ct_id": ct.tooth_id,
#                     "nt_id": nt.tooth_id,
#                     "pitch_mm": abs(ct.grinding_point[1] - nt.grinding_point[1]) / pixels_per_mm,
#                     "depth_mm": abs(ct.grinding_point[0] - ct.bottom_valley[0]) / pixels_per_mm,
#                     "_ct_tip": ct.grinding_point,
#                     "_nt_tip": nt.grinding_point,
#                 }

#         if not closest:
#             return None

#         sharpness = self._groove_sharpness_score(
#             closest.pop("_ct_tip"),
#             closest.pop("_nt_tip"),
#         )
#         closest["sharpness_score"] = round(sharpness, 2) if sharpness is not None else None
#         closest["is_sharp"] = (sharpness is not None and sharpness >= sharp_threshold)
#         return closest

#     def measure_grinder_valley_depth(self):
#         """
#         Measure the groove depth of the valley DIRECTLY ACROSS from the grinder
#         tip — the tooth that was just cut. Depth is the SAME calculation as
#         detection depth: average X of the two neighbouring tooth tips minus the
#         valley bottom X. Only this one valley — no averaging across the blade.
#         Returns dict or None.
#         """
#         global pixels_per_mm
#         grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
#         if not grinder_tip or len(self.teeth_profiles) < 2:
#             return None
#         deduped = _dedupe_clusters(self.teeth_profiles, y_threshold_mm=1.0,
#                                    grinder_tip=grinder_tip)
#         if len(deduped) < 2:
#             return None
#         gt_y = grinder_tip[1]
#         best, best_d = None, float('inf')
#         for i in range(len(deduped) - 1):
#             ct, nt = deduped[i], deduped[i + 1]
#             valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2.0
#             d = abs(valley_y - gt_y)
#             if d < best_d:
#                 best_d, best = d, (ct, nt)
#         if best is None:
#             return None
#         ct, nt = best
#         avg_tip_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2.0
#         valley_x  = ct.bottom_valley[0]
#         depth_mm  = abs(avg_tip_x - valley_x) / pixels_per_mm
#         valley_y  = (ct.grinding_point[1] + nt.grinding_point[1]) / 2.0
#         return {
#             'depth_mm':          round(float(depth_mm), 3),
#             'valley_id':         f'{ct.tooth_id}-{nt.tooth_id}',
#             'valley_x_px':       int(valley_x),
#             'valley_y_px':       int(valley_y),
#             'grinder_offset_px': int(best_d),
#         }

#     def analyze_frame(self, use_stored_grinder=True):
#         try:
#             self.preprocess_image();
#             self.detect_blade_and_grinder()
#             if self.blade_edge_points is None or len(self.blade_edge_points) < 20: return None
#             stored = get_current_grinder_tip()
#             if use_stored_grinder and stored is not None:
#                 self.grinder_tip = stored
#             elif self.grinder_tip is not None:
#                 set_current_grinder_tip(self.grinder_tip)
#             self.teeth_profiles = self.extract_tooth_profiles()
#             if len(self.teeth_profiles) > 0 and self.grinder_tip:
#                 return self._generate_coordinates()
#             return None
#         except Exception as e:
#             print(f"Analysis error: {e}");import traceback;traceback.print_exc();return None


#     def _generate_coordinates(self):
#         """
#         NoSine / deduped method.
#           1. Dedupe detected tops (keep the TALLEST per cluster), pitch-relative threshold.
#           2. Fit the LINEAR ENVELOPE (+ measured pitch + A_mean) via apply_sine_correction
#              on the deduped tops. The envelope smooths the lateral/X tip line; the pitch is
#              MEASURED from the real tops (datasheet pitch is only a safeguard).
#           3. Each valley = a reliable top anchored + 1/2 measured pitch toward the grinder
#              (NOT the midpoint of two raw tops). Valley X from the envelope, depth = A_mean.
#           4. Pick the valley nearest the grinder (>= 0.5 mm above it).
#         The sine WAVE is never used for any value. No off-pitch rejection (yet).
#         """
#         global pixels_per_mm, pitch_mm, last_overlay_pts
#         if len(self.teeth_profiles) < 2:
#             return None
#         grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
#         if not grinder_tip:
#             return None
#         gt_x, gt_y = grinder_tip[0], grinder_tip[1]

#         # Datasheet pitch is only a SAFEGUARD; default 2 mm if nothing set.
#         nominal_pitch_mm = pitch_mm if pitch_mm > 0 else 2.0

#         # 1. Dedupe tops — keep the tallest per cluster, merge anything < 0.5 pitch apart.
#         deduped = _dedupe_clusters(
#             self.teeth_profiles,
#             y_threshold_mm=0.5 * nominal_pitch_mm,
#             select='tallest',
#         )
#         if len(deduped) < 2:
#             return None

#         # 2. Linear envelope + measured pitch + A_mean (from the deduped real tops).
#         corrected_apexes, _corrected_valleys, info, _above = apply_sine_correction(
#             deduped, grinder_tip, pixels_per_mm, nominal_pitch_mm,
#             blade_edge_points=self.blade_edge_points,
#             min_teeth_for_fit=2, baseline_poly_degree=1,
#         )
#         if not info or not corrected_apexes or len(corrected_apexes) < 2:
#             return None
#         f_envelope        = np.poly1d(info['poly_coeffs'])
#         measured_pitch_px = float(info['pitch_px'])              # from real tops, safeguarded to nominal
#         depth_x_mm        = abs(float(info['A_mean'])) / pixels_per_mm

#         apex_ys = sorted(float(a[1]) for a in corrected_apexes)

#         # 3. Valley = each top + 1/2 measured pitch (anchor to the upper/reliable tooth),
#         #    lateral X from the envelope.   4. Pick the nearest valley above the grinder.
#         cand_valleys = []
#         closest_valley = None
#         min_distance = float('inf')
#         for a_y in apex_ys:
#             v_y = a_y + 0.5 * measured_pitch_px
#             v_x = float(f_envelope(v_y))
#             cand_valleys.append((v_x, v_y))
#             move_x_mm = (gt_x - v_x) / pixels_per_mm
#             move_y_mm = (gt_y - v_y) / pixels_per_mm
#             if move_y_mm < 0.5:
#                 continue
#             dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
#             if dist < min_distance:
#                 min_distance = dist
#                 n_above = sum(1 for ay in apex_ys if ay < v_y)
#                 closest_valley = {
#                     'valley_x': v_x,
#                     'valley_y': v_y,
#                     'move_x_mm': move_x_mm,
#                     'move_y_mm': move_y_mm,
#                     'depth_x_mm': depth_x_mm,
#                     'between_teeth': f"{n_above}-{n_above + 1}",
#                     'distance_mm': dist,
#                 }

#         # Store deduped tops + candidate valleys for the live overlay (debug detection).
#         last_overlay_pts = {
#             'apexes':  [(int(t.grinding_point[0]), int(t.grinding_point[1])) for t in deduped],
#             'valleys': [(int(vx), int(vy)) for (vx, vy) in cand_valleys],
#         }

#         if not closest_valley:
#             return None

#         return {
#             'valley_id': closest_valley['between_teeth'],
#             'x_mm':      round(float(closest_valley['move_y_mm']), 2),
#             'y_mm':      round(float(closest_valley['move_x_mm']), 2),
#             'depth_x_mm': round(float(closest_valley['depth_x_mm']), 2),
#             'valley_x_px': int(closest_valley['valley_x']),
#             'valley_y_px': int(closest_valley['valley_y']),
#             'grinder_tip_x_px': int(gt_x),
#             'grinder_tip_y_px': int(gt_y),
#             'num_teeth':  int(len(deduped)),
#             'distance_mm': round(float(closest_valley['distance_mm']), 2),
#             'status': 1,
#             'all_valleys': [],
#             'sine_fit': False,
#         }


# # ── Inspection helpers ────────────────────────────────────────────
# def _write_inspection_log(event_log, samples, blurry_samples):
#     from datetime import datetime
#     ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = 'inspection_log_' + ts + '.txt'
#     hdr = '{:>4}  {:>8}  {:>10}  {:>10}  {:>8}\n'.format('#','angle','depth_mm','pitch_mm','sharpness')
#     def row(s, idx):
#         return '{:>4}  {:>8.2f}  {:>10.4f}  {:>10.4f}  {:>8}\n'.format(
#             idx, s['angle'], s['depth_mm'], s['pitch_mm'], s['sharpness'])
#     try:
#         with open(filename, 'w') as f:
#             f.write('MKA Blade Inspection Log -- ' + datetime.now().isoformat() + '\n')
#             f.write('=' * 70 + '\n\n')
#             f.write('EVENT LOG\n---------\n')
#             for line in event_log:
#                 f.write(line + '\n')
#             f.write('\nSHARP SAMPLES (blur >= threshold)\n')
#             f.write('-' * 34 + '\n')
#             if samples:
#                 f.write(hdr)
#                 for idx, s in enumerate(samples, 1):
#                     f.write(row(s, idx))
#             else:
#                 f.write('None.\n')
#             f.write('\nBLURRY SAMPLES (blur < threshold)\n')
#             f.write('-' * 34 + '\n')
#             if blurry_samples:
#                 f.write(hdr)
#                 for idx, s in enumerate(blurry_samples, 1):
#                     f.write(row(s, idx))
#             else:
#                 f.write('None.\n')
#             f.write('\nSUMMARY\n-------\n')
#             f.write('Sharp accepted : {}\n'.format(len(samples)))
#             f.write('Blurry recorded: {}\n'.format(len(blurry_samples)))
#             if blurry_samples:
#                 t = blurry_samples[0]
#                 f.write('\nFIRST BLUR TRANSITION (result)\n')
#                 f.write('Angle      : {} deg\n'.format(t['angle']))
#                 f.write('Depth      : {} mm\n'.format(t['depth_mm']))
#                 f.write('Pitch      : {} mm\n'.format(t['pitch_mm']))
#                 f.write('Sharpness : {}\n'.format(t['sharpness']))
#             elif samples:
#                 t = samples[0]
#                 f.write('\nFIRST SHARP SAMPLE (no transition found)\n')
#                 f.write('Angle      : {} deg\n'.format(t['angle']))
#                 f.write('Depth      : {} mm\n'.format(t['depth_mm']))
#                 f.write('Pitch      : {} mm\n'.format(t['pitch_mm']))
#                 f.write('Sharpness : {}\n'.format(t['sharpness']))
#         print('Inspection log written: ' + filename)
#         return filename
#     except Exception as e:
#         print('Could not write inspection log: ' + str(e))
#         return None



# # ── Inspection background logic ───────────────────────────────────────────────
# def _run_inspection():
#     """
#     Records (angle, depth_mm, pitch_mm, sharpness) every time REG 142 changes
#     by ≥ ANGLE_STEP°. No sharpness threshold — finalises by selecting the
#     sample with the highest sharpness score.
#     """
#     global inspection_active, last_inspection_summary
#     global last_frame, modbus_client, last_robot_angle

#     ANGLE_STEP = 0.1   # minimum angular change (°) before taking a new sample

#     samples            = []
#     event_log          = []
#     last_sampled_angle = None

#     def _evt(tag, msg):
#         ts   = time.strftime('%H:%M:%S')
#         line = f"[{ts}] {tag:5s} {msg}"
#         event_log.append(line)
#         print(line)

#     _evt('START', 'Calibration inspection recording started (angle-change gated, peak-sharpness mode)')

#     while inspection_active:
#         try:
#             # ── 1. Read current blade angle from REG 142 ──────────────────────
#             angle = last_robot_angle
#             if modbus_client and modbus_client.connected:
#                 val = modbus_client.read_robot_angle()
#                 if val is not None:
#                     angle = val
#                     last_robot_angle = angle

#             # ── 2. Skip if angle hasn't changed enough ────────────────────────
#             if last_sampled_angle is not None and abs(angle - last_sampled_angle) < ANGLE_STEP:
#                 time.sleep(0.05)
#                 continue

#             with camera_lock:
#                 if last_frame is None:
#                     time.sleep(0.05)
#                     continue
#                 frame = last_frame.copy()

#             analyzer = SerratedBladeAnalyzer(frame)
#             analyzer.preprocess_image()
#             analyzer.detect_blade_and_grinder()
#             stored = get_current_grinder_tip()
#             if stored: analyzer.grinder_tip = stored
#             analyzer.teeth_profiles = analyzer.extract_tooth_profiles()
#             stats = analyzer._get_closest_pair_stats()  # threshold no longer matters

#             if not stats:
#                 _evt('SKIP ', 'angle={:.2f}°  — no tooth pair detected'.format(angle))
#             elif stats['sharpness_score'] is None:
#                 _evt('SKIP ', 'angle={:.2f}°  — sharpness unavailable'.format(angle))
#             else:
#                 last_sampled_angle = angle
#                 samples.append({
#                     'angle':     round(angle, 2),
#                     'depth_mm':  round(stats['depth_mm'], 4),
#                     'pitch_mm':  round(stats['pitch_mm'], 4),
#                     'sharpness': stats['sharpness_score'],
#                 })
#                 _evt('OK   ', 'angle={:.2f}°  depth={:.4f}mm  pitch={:.4f}mm  sharp={}'.format(
#                               angle, stats['depth_mm'], stats['pitch_mm'], stats['sharpness_score']))
#                 live = samples[-1]
#                 last_inspection_summary = {
#                     'status':         'running',
#                     'samples':        len(samples),
#                     'live_angle':     live['angle'],
#                     'live_depth':     live['depth_mm'],
#                     'live_pitch':     live['pitch_mm'],
#                     'live_sharpness': live['sharpness'],
#                     'event_log':      list(event_log),
#                 }

#         except Exception as e:
#             _evt('ERR  ', 'sample error: {}'.format(e))
#         time.sleep(0.1)

#     # ── Finalise: pick the sample with the highest sharpness score ───────────
#     if samples:
#         peak = max(samples, key=lambda s: s['sharpness'])
#         _evt('DONE ', 'peak_sharpness: angle={}°  depth={}mm  pitch={}mm  sharp={}  '
#                       'samples={}'.format(
#                       peak['angle'], peak['depth_mm'], peak['pitch_mm'],
#                       peak['sharpness'], len(samples)))
#         last_inspection_summary = {
#             'status':         'complete',
#             'samples':        len(samples),
#             'peak_angle':     peak['angle'],
#             'peak_depth_mm':  peak['depth_mm'],
#             'peak_pitch_mm':  peak['pitch_mm'],
#             'peak_sharpness': peak['sharpness'],
#             'event_log':      list(event_log),
#             'curve':          samples,
#         }
#     else:
#         _evt('DONE ', 'no samples collected')
#         last_inspection_summary = {
#             'status':    'error',
#             'message':   'No samples collected',
#             'event_log': list(event_log),
#         }

#     _write_inspection_log(event_log, samples, [])


# def _teeth_inspect_watcher():
#     """
#     Polls REG 141 (calibration flag) and REG 142 (robot angle) every 150 ms.
#     Rising edge  (0→1): start _run_inspection thread.
#     Falling edge (1→0): set inspection_active=False so thread exits cleanly.
#     """
#     global client_connected, modbus_client, inspection_active
#     global last_inspection_summary, last_teeth_inspect_reg, last_robot_angle

#     while client_connected:
#         try:
#             if modbus_client and modbus_client.connected:
#                 val   = modbus_client.read_teeth_inspect()
#                 angle = modbus_client.read_robot_angle()
#                 if angle is not None:
#                     last_robot_angle = angle
#                 if val is not None:
#                     if val and not last_teeth_inspect_reg:
#                         # Rising edge → start
#                         if not inspection_active and camera_active and last_frame is not None:
#                             print("🔍 REG 141=1 — starting calibration inspection")
#                             last_inspection_summary = {'status':'running','samples':0,
#                                                        'live_angle':0,'live_depth':0,'live_pitch':0}
#                             inspection_active = True
#                             Thread(target=_run_inspection, daemon=True).start()
#                     elif not val and last_teeth_inspect_reg:
#                         # Falling edge → stop
#                         if inspection_active:
#                             print("🔍 REG 141=0 — stopping inspection")
#                             inspection_active = False
#                     last_teeth_inspect_reg = val
#         except Exception:
#             pass
#         time.sleep(0.15)


# # ── Modbus client ─────────────────────────────────────────────────────────────
# class BladeDataModbusClient:
#     """
#     Register map:
#       128  REG_BAY_ID         PC→Robot
#       129  REG_GRINDER_ID     PC→Robot
#       130  REG_ANGLE          PC→Robot  ×10
#       131  REG_DEPTH          PC→Robot  ×100
#       132  REG_LENGTH         PC→Robot  mm
#       133  REG_CONFIG_VERSION PC→Robot
#       134  REG_DETECTION_X    PC→Robot  ×10 signed
#       135  REG_DETECTION_Y    PC→Robot  ×10 signed
#       136  REG_STATUS         PC→Robot  0=none 1=ok 2=err
#       137  REG_START          PC→Robot  1=start 0=idle
#       138  REG_GRINDER_READY  Robot→PC  1=at position
#       139  REG_GRIND_START    PC→Robot  1=grind
#       140  REG_ESTOP          PC→Robot  1=HALT
#       141  REG_TEETH_INSPECT  Robot→PC  1=record 0=stop
#       142  REG_ROBOT_ANGLE    Robot→PC  ×10 signed degrees
#       143  REG_DETECTION_DEPTH PC→Robot  ×100 signed mm  (valley depth between adjacent teeth)
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
#     REG_TEETH_INSPECT  = 141
#     REG_ROBOT_ANGLE    = 142
#     REG_DETECTION_DEPTH = 143  # PC→Robot, X-depth of valley between teeth (×100, signed)
#     REG_POST_CUT_DEPTH  = 150  # PC→Robot, measured groove depth after cut (×100 signed)

#     STATUS_NO_TEETH = 0; STATUS_TEETH_OK = 1; STATUS_ERROR = 2

#     def __init__(self, host='172.24.9.20', port=502, unit=1):
#         self.host=host; self.port=port; self.unit=unit
#         self.client=ModbusTcpClient(host, port=port); self.connected=False

#     def _s16(self, v): return v if v < 32768 else v - 65536

#     def connect(self):
#         if self.client.connect():
#             self.connected=True; print(f"✓ Connected to robot at {self.host}:{self.port}"); return True
#         self.connected=False; print(f"✗ Could not connect to {self.host}:{self.port}"); return False

#     def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
#         if not self.connected: return None
#         values=[int(bay_id),int(grinder_id),int(angle*10),int(depth*100),int(length),int(config_version)]
#         result=self.client.write_registers(address=self.REG_BAY_ID,values=values)
#         if not result.isError(): print(f"✓ Config written")
#         return result

#     def write_detection(self, x_mm, y_mm, status, depth_mm=0.0):
#         if not self.connected: return None

#         # Write depth FIRST (REG 143) so it's stable before STATUS goes high
#         depth_val = int(depth_mm * 100)
#         depth_u16 = depth_val if depth_val >= 0 else 65536 + depth_val
#         self.client.write_register(address=self.REG_DETECTION_DEPTH, value=depth_u16)

#         # Then write X/Y/STATUS contiguously (REG 134-136)
#         x_val = int(x_mm * 100);
#         y_val = int(y_mm * 100)
#         x_u16 = x_val if x_val >= 0 else 65536 + x_val
#         y_u16 = y_val if y_val >= 0 else 65536 + y_val
#         result = self.client.write_registers(address=self.REG_DETECTION_X,
#                                              values=[x_u16, y_u16, int(status)])
#         if not result.isError():
#             print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm "
#                   f"depth={depth_mm:.2f}mm status={status}")
#         return result

#     def write_post_cut_depth(self, depth_mm):
#         if not self.connected: return None
#         val = int(depth_mm * 100)
#         u16 = val if val >= 0 else 65536 + val
#         r = self.client.write_register(address=self.REG_POST_CUT_DEPTH, value=u16)
#         if not r.isError():
#             print(f"✓ Post-cut depth → REG 150: {depth_mm:.2f}mm")
#         return r

#     def start_loop(self):
#         if not self.connected: return None
#         r=self.client.write_register(address=self.REG_START,value=1)
#         if not r.isError(): print("✓ START=1"); return r

#     def stop_loop(self):
#         if not self.connected: return None
#         r=self.client.write_register(address=self.REG_START,value=0)
#         if not r.isError(): print("✓ START=0"); return r

#     def send_grind_start(self):
#         if not self.connected: return None
#         r=self.client.write_register(address=self.REG_GRIND_START,value=1)
#         if not r.isError(): print("✓ GRIND_START=1"); return r

#     def emergency_stop(self):
#         if not self.connected: return None
#         self.client.write_register(address=self.REG_START,value=0)
#         self.client.write_register(address=self.REG_GRIND_START,value=0)
#         r=self.client.write_register(address=self.REG_ESTOP,value=1)
#         if not r.isError(): print("🚨 EMERGENCY STOP SENT")
#         else: print(f"✗ E-STOP failed: {r}")
#         return r

#     def clear_estop(self):
#         if not self.connected: return None
#         r=self.client.write_register(address=self.REG_ESTOP,value=0)
#         if not r.isError(): print("✓ E-STOP cleared"); return r

#     def read_teeth_inspect(self):
#         if not self.connected: return None
#         r=self.client.read_holding_registers(address=self.REG_TEETH_INSPECT,count=1)
#         if r.isError(): return None
#         return bool(r.registers[0])

#     def read_robot_angle(self):
#         """REG 142 — blade angle ×10 signed → float degrees."""
#         if not self.connected: return None
#         r=self.client.read_holding_registers(address=self.REG_ROBOT_ANGLE,count=1)
#         if r.isError(): return None
#         return self._s16(r.registers[0]) / 10.0

#     def read_all_status(self):
#         """Read REG 134–143 (10 registers) in one shot."""
#         if not self.connected: return None
#         r = self.client.read_holding_registers(address=self.REG_DETECTION_X, count=10)
#         if r.isError(): return None
#         reg = r.registers
#         return {
#             'detection_x_mm': self._s16(reg[0]) / 100.0,
#             'detection_y_mm': self._s16(reg[1]) / 100.0,
#             'status': reg[2],
#             'start': reg[3],
#             'grinder_ready': reg[4],
#             'grind_start': reg[5],
#             'estop': reg[6],
#             'teeth_inspect': bool(reg[7]),
#             'robot_angle': self._s16(reg[8]) / 10.0,
#             'detection_depth_mm': self._s16(reg[9]) / 100.0,
#         }

#     def close(self):
#         self.client.close(); self.connected=False; print("✓ Modbus connection closed")


# # ── Flask routes ──────────────────────────────────────────────────────────────
# @app.route('/')
# def index():
#     try:
#         with open('robot_control_dashboard.html', 'r', encoding='utf-8') as f:
#             return f.read()
#     except FileNotFoundError:
#         import os
#         return f"<h1>robot_control_dashboard.html not found</h1><p>cwd:{os.getcwd()}</p>", 404

# @app.route('/api/blade/<blade_id>',methods=['GET'])
# def get_blade(blade_id):
#     import sqlite3
#     try:
#         conn=sqlite3.connect("blade_database.sqlite"); conn.row_factory=sqlite3.Row
#         cur=conn.cursor(); cur.execute("SELECT * FROM blades WHERE bade_id=?",(blade_id,))
#         row=cur.fetchone(); conn.close()
#         if row: return jsonify({"success":True,"blade":dict(row)})
#         return jsonify({"success":False,"message":f"Blade '{blade_id}' not found"})
#     except Exception as e: return jsonify({"success":False,"message":str(e)})

# @app.route('/api/connect',methods=['POST'])
# def connect():
#     global modbus_client,client_connected
#     data=request.json; host=data.get('host','172.24.9.20')
#     port=int(data.get('port',502)); unit=int(data.get('unit',1))
#     try:
#         modbus_client=BladeDataModbusClient(host=host,port=port,unit=unit)
#         if modbus_client.connect():
#             client_connected=True
#             Thread(target=_teeth_inspect_watcher,daemon=True).start()
#             return jsonify({'success':True,'message':f'Connected to robot at {host}:{port}'})
#         client_connected=False
#         return jsonify({'success':False,'message':f'Failed to connect to {host}:{port}'}),500
#     except Exception as e:
#         client_connected=False; return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/disconnect',methods=['POST'])
# def disconnect():
#     global modbus_client,client_connected,inspection_active
#     if modbus_client:
#         try:
#             inspection_active=False
#             modbus_client.close(); modbus_client=None; client_connected=False
#             return jsonify({'success':True,'message':'Disconnected'})
#         except Exception as e: return jsonify({'success':False,'message':str(e)}),500
#     return jsonify({'success':True,'message':'Already disconnected'})

# @app.route('/api/status',methods=['GET'])
# def get_status():
#     return jsonify({'connected':client_connected and modbus_client is not None})

# @app.route('/api/configuration', methods=['POST'])
# def send_configuration():
#     global current_grinder_id
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success': False, 'message': 'Not connected'}), 400
#     data = request.json
#     try:
#         new_gid = int(data.get('grinder_id'))
#         result = modbus_client.write_configuration(
#             bay_id=int(data.get('bay_id')), grinder_id=new_gid,
#             angle=float(data.get('angle')), depth=float(data.get('depth')),
#             length=int(data.get('length')), config_version=int(data.get('config_version')))
#         if not (result and not result.isError()):
#             return jsonify({'success': False, 'message': 'Failed to send configuration'}), 500

#         global last_commanded_depth_mm
#         last_commanded_depth_mm = float(data.get('depth'))

#         switched = False
#         if new_gid != current_grinder_id:
#             print(f"🔄 Auto-switching camera: grinder {current_grinder_id} → {new_gid}")
#             was_active = camera_active
#             if was_active:
#                 _stop_camera_internal()
#             current_grinder_id = new_gid
#             if was_active:
#                 ok, msg = _start_camera_internal(new_gid)
#                 if not ok:
#                     return jsonify({'success': False,
#                                     'message': f'Config sent but camera switch failed: {msg}'}), 500
#             switched = True
#         return jsonify({'success': True,
#                         'message': f'Configuration sent (REG 128-133)'
#                                    f'{" + camera switched" if switched else ""}',
#                         'grinder_id': current_grinder_id,
#                         'camera_switched': switched})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# @app.route('/api/detection',methods=['POST'])
# def send_detection():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     data=request.json
#     try:
#         result=modbus_client.write_detection(x_mm=float(data.get('x_mm')),y_mm=float(data.get('y_mm')),status=int(data.get('status',1)))
#         if result and not result.isError(): return jsonify({'success':True,'message':'Detection data sent'})
#         return jsonify({'success':False,'message':'Failed to send detection'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/start',methods=['POST'])
# def start_robot():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     try:
#         result=modbus_client.start_loop()
#         if result and not result.isError(): return jsonify({'success':True,'message':'Loop started (START=1)'})
#         return jsonify({'success':False,'message':'Failed to start loop'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/stop',methods=['POST'])
# def stop_robot():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     try:
#         result=modbus_client.stop_loop()
#         if result and not result.isError(): return jsonify({'success':True,'message':'Loop stopped (START=0)'})
#         return jsonify({'success':False,'message':'Failed to stop loop'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/estop',methods=['POST'])
# def emergency_stop():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected to robot'}),400
#     try:
#         result=modbus_client.emergency_stop()
#         if result and not result.isError(): return jsonify({'success':True,'message':'EMERGENCY STOP — REG 140=1'})
#         return jsonify({'success':False,'message':'Failed to send E-STOP'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/estop/clear',methods=['POST'])
# def clear_estop():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     try:
#         result=modbus_client.clear_estop()
#         if result and not result.isError(): return jsonify({'success':True,'message':'E-STOP cleared (REG 140=0)'})
#         return jsonify({'success':False,'message':'Failed to clear E-STOP'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/grind_start',methods=['POST'])
# def grind_start():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     data=request.json or {}
#     try:
#         if 'x_mm' in data and 'y_mm' in data:
#             modbus_client.write_detection(
#                 x_mm=float(data['x_mm']),
#                 y_mm=float(data['y_mm']),
#                 status=int(data.get('status', 1)),
#                 depth_mm=float(data.get('depth_mm', 0.0)),
#             )
#             global last_grind_depth_mm
#             last_grind_depth_mm = float(data.get('depth_mm', 0.0))
#         result=modbus_client.send_grind_start()
#         if result and not result.isError(): return jsonify({'success':True,'message':'GRIND_START=1 sent'})
#         return jsonify({'success':False,'message':'Failed to send GRIND_START'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/robot_registers',methods=['GET'])
# def robot_registers():
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     try:
#         status=modbus_client.read_all_status()
#         if status: return jsonify({'success':True,'registers':status})
#         return jsonify({'success':False,'message':'Failed to read registers'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/camera/ready', methods=['GET'])
# def camera_ready():
#     return jsonify({'ready': camera_active and last_frame is not None})

# # ── Inspection endpoints ──────────────────────────────────────────────────────
# @app.route('/api/inspection/start',methods=['POST'])
# def start_inspection():
#     global inspection_active,last_inspection_summary
#     if not camera_active or last_frame is None:
#         return jsonify({'success':False,'message':'Camera not active'}),400
#     if inspection_active:
#         return jsonify({'success':False,'message':'Inspection already running'}),400
#     last_inspection_summary={'status':'running','samples':0,'live_angle':0,'live_depth':0,'live_pitch':0}
#     inspection_active=True
#     Thread(target=_run_inspection,daemon=True).start()
#     return jsonify({'success':True,'message':'Inspection started'})

# @app.route('/api/inspection/stop',methods=['POST'])
# def stop_inspection():
#     global inspection_active
#     if not inspection_active:
#         return jsonify({'success':False,'message':'Inspection not running'}),400
#     inspection_active=False
#     return jsonify({'success':True,'message':'Inspection stopping…'})

# @app.route('/api/inspection/status',methods=['GET'])
# def inspection_status_route():
#     return jsonify({'active':inspection_active,'result':last_inspection_summary,'current_angle':last_robot_angle})


# # ── PySpin system management (KEEP ALIVE for the whole process) ───────────────
# # System.GetInstance() + GetCameras() triggers a full Spinnaker device
# # discovery, which on Windows can take 30-40+ seconds (it probes every
# # network interface). We therefore do it ONCE and keep both the System and
# # the CameraList alive. Camera switches only DeInit/Init the camera object.

# def _get_pyspin_system(refresh=False):
#     """
#     Initialise Spinnaker once and keep it alive for the whole process.
#     refresh=True re-enumerates the camera list (only safe while no camera
#     is acquiring) — used if cameras were plugged in after startup.
#     """
#     global pyspin_system, pyspin_cam_list
#     if pyspin_system is None:
#         print("🎥 Initialising Spinnaker system (one-time discovery — may take a while)…")
#         t0 = time.perf_counter()
#         pyspin_system = PySpin.System.GetInstance()
#         pyspin_cam_list = pyspin_system.GetCameras()
#         print(f"✓ Spinnaker ready in {time.perf_counter()-t0:.1f}s "
#               f"({pyspin_cam_list.GetSize()} camera(s) found)")
#     elif refresh and not camera_active:
#         try:
#             pyspin_cam_list.Clear()
#         except Exception:
#             pass
#         pyspin_cam_list = pyspin_system.GetCameras()
#         print(f"✓ Camera list refreshed ({pyspin_cam_list.GetSize()} camera(s))")
#     return pyspin_system, pyspin_cam_list


# @atexit.register
# def _release_pyspin():
#     """Release Spinnaker cleanly when the process exits."""
#     global pyspin_cam, pyspin_cam_list, pyspin_system
#     try:
#         if pyspin_cam:
#             try:
#                 pyspin_cam.EndAcquisition()
#             except Exception:
#                 pass
#             try:
#                 pyspin_cam.DeInit()
#             except Exception:
#                 pass
#             pyspin_cam = None
#         if pyspin_cam_list is not None:
#             try:
#                 pyspin_cam_list.Clear()
#             except Exception:
#                 pass
#             pyspin_cam_list = None
#         if pyspin_system is not None:
#             try:
#                 pyspin_system.ReleaseInstance()
#             except Exception:
#                 pass
#             pyspin_system = None
#         print("✓ Spinnaker released")
#     except Exception:
#         pass


# # ── Camera ────────────────────────────────────────────────────────────────────
# def _stop_camera_internal():
#     """Stop capture and DeInit the camera. The PySpin System and CameraList
#     are intentionally KEPT ALIVE so the next start/switch is fast."""
#     global pyspin_cam, camera_active, camera_thread, last_frame
#     camera_active = False
#     th = camera_thread
#     camera_thread = None
#     if th and th.is_alive():
#         th.join(timeout=2.0)
#     with camera_lock:
#         if pyspin_cam:
#             try:
#                 pyspin_cam.EndAcquisition()
#             except Exception:
#                 pass
#             try:
#                 pyspin_cam.DeInit()
#             except Exception:
#                 pass
#             pyspin_cam = None
#         last_frame = None
#         # NOTE: system + cam_list intentionally kept alive (see _get_pyspin_system)


# def _start_camera_internal(grinder_id):
#     """Initialise and start PySpin camera for the given grinder id. Returns (ok, msg).
#     Uses the persistent system/camera list — no Spinnaker re-discovery."""
#     global pyspin_cam, camera_active, camera_thread
#     if camera_active:
#         return False, 'Camera already running (stop first)'
#     serial = CAMERA_SERIALS.get(grinder_id, '')
#     print(f"🎥 Starting FLIR camera for grinder {grinder_id} "
#           f"(serial='{serial or 'auto-index ' + str(grinder_id - 1)}')")

#     system, cam_list = _get_pyspin_system()
#     n = cam_list.GetSize()
#     if n == 0:
#         # Maybe cameras were plugged in after startup — refresh the list once
#         system, cam_list = _get_pyspin_system(refresh=True)
#         n = cam_list.GetSize()
#         if n == 0:
#             return False, 'No FLIR cameras detected'

#     selected = None
#     selected_idx = None
#     if serial:
#         for i in range(n):
#             cam = cam_list[i]
#             cam_sn = ''
#             try:
#                 tldev = cam.GetTLDeviceNodeMap()
#                 sn_node = PySpin.CStringPtr(tldev.GetNode('DeviceSerialNumber'))
#                 if PySpin.IsReadable(sn_node):
#                     cam_sn = sn_node.GetValue()
#                 del sn_node, tldev
#             except Exception:
#                 cam_sn = ''
#             if cam_sn == serial:
#                 selected_idx = i
#                 del cam
#                 break
#             del cam
#         if selected_idx is None:
#             return False, f"Camera with serial '{serial}' not found for grinder {grinder_id}"
#         selected = cam_list[selected_idx]
#     else:
#         idx = grinder_id - 1
#         if not (0 <= idx < n):
#             return False, f'Camera index {idx} out of range (have {n})'
#         selected = cam_list[idx]

#     try:
#         pyspin_cam = selected
#         pyspin_cam.Init()
#         _configure_pyspin_camera()
#         pyspin_cam.BeginAcquisition()
#     except PySpin.SpinnakerException as ex:
#         # Clean up the half-initialised camera so a retry is possible
#         try:
#             if pyspin_cam:
#                 pyspin_cam.DeInit()
#         except Exception:
#             pass
#         pyspin_cam = None
#         return False, f'Camera init failed: {ex}'

#     camera_active = True
#     camera_thread = Thread(target=_camera_capture_thread, daemon=True)
#     camera_thread.start()
#     print(f"✓ FLIR camera started for grinder {grinder_id}")
#     return True, f'Camera started for grinder {grinder_id}'


# @app.route('/api/camera/start', methods=['POST'])
# def start_camera():
#     global current_grinder_id
#     data = request.json or {}
#     if 'grinder_id' in data:
#         current_grinder_id = int(data['grinder_id'])
#     ok, msg = _start_camera_internal(current_grinder_id)
#     if ok:
#         return jsonify({'success': True, 'message': msg, 'grinder_id': current_grinder_id})
#     return jsonify({'success': False, 'message': msg}), 500


# @app.route('/api/camera/stop', methods=['POST'])
# def stop_camera():
#     try:
#         _stop_camera_internal()
#         return jsonify({'success': True, 'message': 'Camera stopped'})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# @app.route('/api/camera/switch', methods=['POST'])
# def switch_camera():
#     """Switch active grinder/camera. Stops + restarts camera if it was running."""
#     global current_grinder_id
#     data = request.json or {}
#     new_gid = int(data.get('grinder_id', current_grinder_id))
#     if new_gid not in CAMERA_SERIALS:
#         return jsonify({'success': False, 'message': f'Unknown grinder_id {new_gid}'}), 400
#     was_active = camera_active
#     if was_active:
#         _stop_camera_internal()
#     current_grinder_id = new_gid
#     if was_active:
#         ok, msg = _start_camera_internal(new_gid)
#         if not ok:
#             return jsonify({'success': False, 'message': f'Switch failed: {msg}',
#                             'grinder_id': new_gid}), 500
#         return jsonify({'success': True, 'grinder_id': new_gid,
#                         'message': f'Switched to grinder {new_gid} (camera restarted)'})
#     return jsonify({'success': True, 'grinder_id': new_gid,
#                     'message': f'Switched to grinder {new_gid} (camera was off)'})

# @app.route('/api/camera/list', methods=['GET'])
# def list_cameras():
#     """Enumerate detected FLIR cameras. Camera must be stopped to (re)enumerate.
#     Uses the persistent Spinnaker system — no full re-discovery."""
#     if camera_active:
#         return jsonify({'success': False, 'message': 'Stop camera first to list devices',
#                         'mapping': CAMERA_SERIALS,
#                         'current_grinder_id': current_grinder_id}), 400
#     try:
#         # Refresh the persistent list (safe: camera is not active)
#         system, cl = _get_pyspin_system(refresh=True)
#         cams = []
#         n = cl.GetSize()
#         for i in range(n):
#             cam = cl[i]
#             sn_value = 'unknown'
#             md_value = 'unknown'
#             try:
#                 tldev = cam.GetTLDeviceNodeMap()
#                 sn = PySpin.CStringPtr(tldev.GetNode('DeviceSerialNumber'))
#                 md = PySpin.CStringPtr(tldev.GetNode('DeviceModelName'))
#                 if PySpin.IsReadable(sn): sn_value = sn.GetValue()
#                 if PySpin.IsReadable(md): md_value = md.GetValue()
#                 # Drop refs to nodes/nodemap BEFORE the camera goes out of scope
#                 del sn, md, tldev
#             except Exception as e:
#                 sn_value = 'error'
#                 md_value = str(e)
#             cams.append({'index': i, 'serial': sn_value, 'model': md_value})
#             # Drop the camera ref every iteration — critical for -1004 fix
#             del cam
#         return jsonify({'success': True, 'cameras': cams,
#                         'mapping': CAMERA_SERIALS,
#                         'current_grinder_id': current_grinder_id})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# @app.route('/api/camera/mapping', methods=['POST'])
# def set_camera_mapping():
#     """Update {grinder_id: serial} map and persist to disk."""
#     global CAMERA_SERIALS
#     data = request.json or {}
#     try:
#         new_map = {int(k): str(v) for k, v in (data.get('mapping') or {}).items()}
#         CAMERA_SERIALS.update(new_map)
#         save_camera_serials()
#         return jsonify({'success': True, 'mapping': CAMERA_SERIALS})
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# @app.route('/api/grinder/current', methods=['GET'])
# def get_current_grinder():
#     return jsonify({'success': True,
#                     'current_grinder_id': current_grinder_id,
#                     'tip': (list(get_current_grinder_tip()) if get_current_grinder_tip() else None),
#                     'all_tips': {k: list(v) for k, v in stored_grinder_tips.items()},
#                     'mapping': CAMERA_SERIALS})

# def _configure_pyspin_camera():
#     global pyspin_cam,camera_config
#     try:
#         nodemap=pyspin_cam.GetNodeMap()
#         node_acq=PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
#         if PySpin.IsWritable(node_acq): node_acq.SetIntValue(node_acq.GetEntryByName('Continuous').GetValue())
#         node_fr_en=PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
#         if PySpin.IsWritable(node_fr_en): node_fr_en.SetValue(True)
#         node_fr=PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
#         if PySpin.IsWritable(node_fr): node_fr.SetValue(min(node_fr.GetMax(),camera_config['frame_rate']))
#         # node_exp_auto=PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
#         # if PySpin.IsWritable(node_exp_auto): node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
#         # node_exp=PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
#         # if PySpin.IsWritable(node_exp): node_exp.SetValue(min(node_exp.GetMax(),camera_config['exposure_time']))
#         node_gain_auto=PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
#         if PySpin.IsWritable(node_gain_auto): node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
#         # node_gain=PySpin.CFloatPtr(nodemap.GetNode('Gain'))
#         # if PySpin.IsWritable(node_gain): node_gain.SetValue(min(node_gain.GetMax(),camera_config['gain']))
#     except PySpin.SpinnakerException as ex: print(f"⚠ Camera config warning: {ex}")


# # ── Camera settings (live tuning: exposure / gain / white balance / fps …) ────
# # These read and write Spinnaker feature nodes on the running camera. Reads are
# # open to viewers; writes go through the control lock (POST is gated) so only the
# # controller can change the shared camera. Node names follow the GenICam SFNC
# # standard used by FLIR/Teledyne cameras.

# def _cam_read_enum(nm, name):
#     try:
#         node = PySpin.CEnumerationPtr(nm.GetNode(name))
#         if not PySpin.IsReadable(node):
#             return None
#         entries = []
#         for i in range(node.GetNumEntries()):
#             e = PySpin.CEnumEntryPtr(node.GetEntryByIndex(i))
#             if PySpin.IsAvailable(e) and PySpin.IsReadable(e):
#                 entries.append(e.GetSymbolic())
#         cur = PySpin.CEnumEntryPtr(node.GetCurrentEntry())
#         return {'value': cur.GetSymbolic() if PySpin.IsReadable(cur) else None,
#                 'entries': entries,
#                 'writable': bool(PySpin.IsWritable(node))}
#     except Exception:
#         return None

# def _cam_read_float(nm, name):
#     try:
#         node = PySpin.CFloatPtr(nm.GetNode(name))
#         if not PySpin.IsReadable(node):
#             return None
#         return {'value': round(float(node.GetValue()), 3),
#                 'min': round(float(node.GetMin()), 3),
#                 'max': round(float(node.GetMax()), 3),
#                 'writable': bool(PySpin.IsWritable(node))}
#     except Exception:
#         return None

# def _cam_read_bool(nm, name):
#     try:
#         node = PySpin.CBooleanPtr(nm.GetNode(name))
#         if not PySpin.IsReadable(node):
#             return None
#         return {'value': bool(node.GetValue()),
#                 'writable': bool(PySpin.IsWritable(node))}
#     except Exception:
#         return None

# def _cam_set_enum(nm, name, symbolic):
#     node = PySpin.CEnumerationPtr(nm.GetNode(name))
#     if not PySpin.IsWritable(node):
#         raise RuntimeError(f'{name} not writable')
#     entry = PySpin.CEnumEntryPtr(node.GetEntryByName(str(symbolic)))
#     if not PySpin.IsReadable(entry):
#         raise RuntimeError(f'{name}={symbolic} not available')
#     node.SetIntValue(entry.GetValue())

# def _cam_set_float(nm, name, value):
#     node = PySpin.CFloatPtr(nm.GetNode(name))
#     if not PySpin.IsWritable(node):
#         raise RuntimeError(f'{name} not writable (is its Auto mode still on?)')
#     v = max(node.GetMin(), min(node.GetMax(), float(value)))
#     node.SetValue(v)

# def _cam_set_bool(nm, name, value):
#     node = PySpin.CBooleanPtr(nm.GetNode(name))
#     if not PySpin.IsWritable(node):
#         raise RuntimeError(f'{name} not writable')
#     node.SetValue(bool(value))

# def _get_camera_settings():
#     global pyspin_cam
#     if not (camera_active and pyspin_cam):
#         return {'success': False, 'message': 'Camera not active'}
#     with camera_lock:
#         try:
#             nm = pyspin_cam.GetNodeMap()
#             return {
#                 'success': True,
#                 'exposure_auto':      _cam_read_enum(nm, 'ExposureAuto'),
#                 'exposure_time':      _cam_read_float(nm, 'ExposureTime'),   # microseconds
#                 'gain_auto':          _cam_read_enum(nm, 'GainAuto'),
#                 'gain':               _cam_read_float(nm, 'Gain'),           # dB
#                 'white_balance_auto': _cam_read_enum(nm, 'BalanceWhiteAuto'),# color cams only
#                 'frame_rate_enable':  _cam_read_bool(nm, 'AcquisitionFrameRateEnable'),
#                 'frame_rate':         _cam_read_float(nm, 'AcquisitionFrameRate'),
#                 'gamma_enable':       _cam_read_bool(nm, 'GammaEnable'),
#                 'gamma':              _cam_read_float(nm, 'Gamma'),
#                 'black_level':        _cam_read_float(nm, 'BlackLevel'),
#                 'grinder_id':         current_grinder_id,
#             }
#         except Exception as e:
#             return {'success': False, 'message': str(e)}

# @app.route('/api/camera/settings', methods=['GET'])
# def camera_settings_get():
#     return jsonify(_get_camera_settings())

# @app.route('/api/camera/settings', methods=['POST'])
# def camera_settings_set():
#     from flask import request
#     global pyspin_cam, camera_config
#     if not (camera_active and pyspin_cam):
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     data = request.json or {}
#     applied, errors = [], []

#     def _try(label, fn):
#         try:
#             fn(); applied.append(label)
#         except Exception as e:
#             errors.append(f'{label}: {e}')

#     def _is_auto(v):
#         return str(v) in ('Continuous', 'Once')

#     with camera_lock:
#         try:
#             nm = pyspin_cam.GetNodeMap()
#         except Exception as e:
#             return jsonify({'success': False, 'message': f'nodemap error: {e}'}), 500

#         # Order matters: set Auto modes BEFORE their manual values, because
#         # ExposureTime/Gain are only writable once their Auto is Off.
#         if 'exposure_auto' in data:
#             _try('exposure_auto', lambda: _cam_set_enum(nm, 'ExposureAuto', data['exposure_auto']))
#         if data.get('exposure_time') is not None and not _is_auto(data.get('exposure_auto')):
#             _try('exposure_time', lambda: _cam_set_float(nm, 'ExposureTime', data['exposure_time']))

#         if 'gain_auto' in data:
#             _try('gain_auto', lambda: _cam_set_enum(nm, 'GainAuto', data['gain_auto']))
#         if data.get('gain') is not None and not _is_auto(data.get('gain_auto')):
#             _try('gain', lambda: _cam_set_float(nm, 'Gain', data['gain']))

#         if 'white_balance_auto' in data:
#             _try('white_balance_auto', lambda: _cam_set_enum(nm, 'BalanceWhiteAuto', data['white_balance_auto']))

#         # Frame rate: must enable manual frame-rate control before setting it.
#         if 'frame_rate_enable' in data:
#             _try('frame_rate_enable', lambda: _cam_set_bool(nm, 'AcquisitionFrameRateEnable', data['frame_rate_enable']))
#         if data.get('frame_rate') is not None and data.get('frame_rate_enable', True):
#             _try('frame_rate', lambda: _cam_set_float(nm, 'AcquisitionFrameRate', data['frame_rate']))

#         if 'gamma_enable' in data:
#             _try('gamma_enable', lambda: _cam_set_bool(nm, 'GammaEnable', data['gamma_enable']))
#         if data.get('gamma') is not None:
#             _try('gamma', lambda: _cam_set_float(nm, 'Gamma', data['gamma']))

#         if data.get('black_level') is not None:
#             _try('black_level', lambda: _cam_set_float(nm, 'BlackLevel', data['black_level']))

#     # Persist frame rate so a camera restart keeps it.
#     if data.get('frame_rate') is not None:
#         try: camera_config['frame_rate'] = float(data['frame_rate'])
#         except Exception: pass

#     fresh = _get_camera_settings()
#     print(f"🎛  Camera settings applied={applied}" + (f" errors={errors}" if errors else ""))
#     return jsonify({'success': len(errors) == 0, 'applied': applied,
#                     'errors': errors, 'settings': fresh})


# def _camera_capture_thread():
#     global pyspin_cam,camera_active,last_frame
#     processor=PySpin.ImageProcessor()
#     processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
#     while camera_active:
#         try:
#             image_result=pyspin_cam.GetNextImage(1000)
#             if not image_result.IsIncomplete():
#                 pf=image_result.GetPixelFormat()
#                 if pf==PySpin.PixelFormat_BGR8: frame=image_result.GetNDArray()
#                 elif pf==PySpin.PixelFormat_Mono8: frame=cv2.cvtColor(image_result.GetNDArray(),cv2.COLOR_GRAY2BGR)
#                 else: frame=processor.Convert(image_result,PySpin.PixelFormat_BGR8).GetNDArray()
#                 frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 with camera_lock: last_frame=frame.copy()
#                 image_result.Release()
#         except PySpin.SpinnakerException: time.sleep(0.01)
#         except Exception as e:
#             if camera_active: print(f"⚠ Frame error: {e}")
#             time.sleep(0.01)

# def draw_detection_overlay(frame,detection_result):
#     overlay=frame.copy()
#     if not detection_result: return overlay
#     try:
#         gt=(detection_result.get('grinder_tip_x_px'),detection_result.get('grinder_tip_y_px'))
#         if gt and gt[0]>0:
#             gti=(int(gt[0]),int(gt[1]))
#             cv2.circle(overlay,gti,12,(0,255,255),3)
#             cv2.putText(overlay,"GRINDER",(gti[0]+20,gti[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
#         vx,vy=detection_result.get('valley_x_px'),detection_result.get('valley_y_px')
#         if vx and vy:
#             vp=(int(vx),int(vy)); cv2.circle(overlay,vp,10,(255,0,255),-1); cv2.circle(overlay,vp,12,(255,255,255),2)
#             cv2.putText(overlay,f"({detection_result.get('x_mm',0):+.1f},{detection_result.get('y_mm',0):+.1f})mm",(vp[0]-35,vp[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
#             if gt and gt[0]>0: cv2.arrowedLine(overlay,vp,(int(gt[0]),int(gt[1])),(0,255,255),2,tipLength=0.02)
#         # NoSine overlay: deduped tops (green stars) + candidate valleys (blue dots)
#         if last_overlay_pts:
#             for (ax, ay) in last_overlay_pts.get('apexes', []):
#                 cv2.drawMarker(overlay, (int(ax), int(ay)), (0, 255, 0),
#                                markerType=cv2.MARKER_STAR, markerSize=12, thickness=2)
#             for (cvx, cvy) in last_overlay_pts.get('valleys', []):
#                 cv2.circle(overlay, (int(cvx), int(cvy)), 5, (255, 128, 0), -1)
#         # Sine envelope is drawn from the fresh per-frame fit in get_camera_frame (gated by the
#         # dashboard 'sine' overlay mode via sine_overlay_enabled), not from the stale loop fit here.
#     except Exception as e: print(f"Overlay error: {e}")
#     return overlay

# @app.route('/api/camera/frame')
# def get_camera_frame():
#     global camera_active,last_frame,last_detection_result
#     if not camera_active or last_frame is None:
#         blank=np.zeros((480,640,3),dtype=np.uint8); _,buf=cv2.imencode('.jpg',blank)
#         return Response(buf.tobytes(),mimetype='image/jpeg')
#     try:
#         with camera_lock: frame=last_frame.copy() if last_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
#         if last_detection_result: frame=draw_detection_overlay(frame,last_detection_result)
#         if sine_overlay_enabled:
#             if live_sine_overlay and (time.time()-live_sine_overlay.get('t',0))<1.0:
#                 try:
#                     draw_sine_overlay(frame,live_sine_overlay['apexes'],
#                                       live_sine_overlay['valleys'],live_sine_overlay['info'])
#                 except Exception as _e:
#                     print(f"[sine_overlay] live draw error: {_e}")
#             else:
#                 hint=('SINE: set pitch > 0 to enable fit' if pitch_mm<=0
#                       else 'SINE: waiting for fit (need grinder tip + teeth)')
#                 cv2.putText(frame,hint,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,128),2)
#         ret,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
#         return Response(buf.tobytes() if ret else np.zeros((480,640,3),dtype=np.uint8),mimetype='image/jpeg')
#     except:
#         blank=np.zeros((480,640,3),dtype=np.uint8); _,buf=cv2.imencode('.jpg',blank)
#         return Response(buf.tobytes(),mimetype='image/jpeg')

# @app.route('/api/camera/capture',methods=['POST'])
# def capture_frame():
#     if not camera_active or last_frame is None:
#         return jsonify({'success':False,'message':'Camera not active'}),400
#     try:
#         with camera_lock: frame=last_frame.copy()
#         filename=f'capture_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
#         cv2.imwrite(filename,frame)   # explicit user action — keep the actual save
#         return jsonify({'success':True,'message':'Captured','filename':filename})
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/detection/analyze',methods=['POST'])
# def analyze_current_frame():
#     global camera_active,last_frame,last_detection_result
#     if not camera_active or last_frame is None:
#         return jsonify({'success':False,'message':'Camera not active'}),400
#     try:
#         with camera_lock: frame_to_process=last_frame.copy()
#         analyzer=SerratedBladeAnalyzer(frame_to_process); result=analyzer.analyze_frame()
#         if result:
#             last_detection_result=result
#             return jsonify({'success':True,'detection':result,'message':f"Valley {result.get('valley_id','N/A')}"})
#         return jsonify({'success':False,'message':'No teeth detected'}),404
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/detection/measure_post_cut', methods=['POST'])
# def measure_post_cut():
#     """Triggered on GRIND_START 1→0. Measures the just-cut valley's depth,
#     writes it to REG 150, and returns measured vs expected for validation."""
#     global last_post_cut_result
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     data = request.json or {}
#     settle_ms = int(data.get('settle_ms', 150))
#     time.sleep(settle_ms / 1000.0)   # let grinder retract / frame settle
#     try:
#         with camera_lock:
#             frame = last_frame.copy()
#         an = SerratedBladeAnalyzer(frame)
#         an.preprocess_image(); an.detect_blade_and_grinder()
#         stored = get_current_grinder_tip()
#         if stored: an.grinder_tip = stored
#         an.teeth_profiles = an.extract_tooth_profiles()
#         meas = an.measure_grinder_valley_depth()
#         if not meas:
#             return jsonify({'success': False,
#                             'message': 'Could not measure post-cut valley'}), 404

#         commanded = last_commanded_depth_mm
#         pre_cut   = last_grind_depth_mm
#         measured  = meas['depth_mm']
#         # Expected FINAL valley depth = commanded depth (REG 131), measured from the
#         # tooth-tip line. The commanded value IS the target total depth, not an
#         # increment on the existing groove — so expected = commanded, NOT pre_cut + commanded.
#         expected  = commanded
#         delta     = (measured - expected) if expected is not None else None

#         if modbus_client and modbus_client.connected:
#             modbus_client.write_post_cut_depth(measured)

#         last_post_cut_result = {
#             'measured_depth_mm':  measured,
#             'commanded_depth_mm': commanded,
#             'pre_cut_depth_mm':   pre_cut,
#             'expected_depth_mm':  round(expected, 3) if expected is not None else None,
#             'delta_mm':           round(delta, 3) if delta is not None else None,
#             'valley_id':          meas['valley_id'],
#             'valley_x_px':        meas['valley_x_px'],
#             'valley_y_px':        meas['valley_y_px'],
#             'ts':                 time.strftime('%H:%M:%S'),
#         }
#         return jsonify({'success': True, 'result': last_post_cut_result})
#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return jsonify({'success': False, 'message': str(e)}), 500

# @app.route('/api/detection/send_auto',methods=['POST'])
# def send_auto_detection():
#     global camera_active,last_frame,modbus_client
#     if not camera_active or last_frame is None:
#         return jsonify({'success':False,'message':'Camera not active'}),400
#     if not modbus_client or not modbus_client.connected:
#         return jsonify({'success':False,'message':'Not connected'}),400
#     try:
#         with camera_lock: frame_to_process=last_frame.copy()
#         analyzer=SerratedBladeAnalyzer(frame_to_process); result=analyzer.analyze_frame()
#         if not result: return jsonify({'success':False,'message':'No teeth detected'}),404
#         modbus_result = modbus_client.write_detection(
#             x_mm=result['x_mm'],
#             y_mm=result['y_mm'],
#             status=result['status'],
#             depth_mm=result.get('depth_x_mm', 0.0),
#         )
#         if modbus_result and not modbus_result.isError():
#             return jsonify({'success':True,'detection':result,'message':f"Valley {result.get('valley_id','N/A')} sent"})
#         return jsonify({'success':False,'message':'Failed to send to robot'}),500
#     except Exception as e: return jsonify({'success':False,'message':str(e)}),500

# @app.route('/api/detection/status',methods=['GET'])
# def get_detection_status():
#     return jsonify({'enabled':detection_enabled,'last_result':last_detection_result,'pixels_per_mm':pixels_per_mm,'pitch_mm':pitch_mm,'sine_overlay':sine_overlay_enabled})


# @app.route('/api/detection/sine_overlay', methods=['GET', 'POST'])
# def set_sine_overlay():
#     global sine_overlay_enabled
#     if request.method == 'POST':
#         data = request.json or {}
#         sine_overlay_enabled = bool(data.get('enabled', False))
#         print(f"[sine_overlay] live envelope {'SHOWN' if sine_overlay_enabled else 'HIDDEN'}")
#     return jsonify({'success': True, 'enabled': sine_overlay_enabled})

# @app.route('/api/detection/pitch', methods=['POST'])
# def set_pitch():
#     global pitch_mm
#     data = request.json or {}
#     val  = data.get('pitch_mm', 0)
#     try:
#         pitch_mm = max(0.0, float(val))
#         msg = (f'Pitch = {pitch_mm:.3f} mm — '
#                f"sine-fit {'ENABLED' if pitch_mm > 0 else 'DISABLED'}")
#         print(f'[pitch] {msg}')
#         return jsonify({'success': True, 'pitch_mm': pitch_mm, 'message': msg})
#     except (TypeError, ValueError) as e:
#         return jsonify({'success': False, 'message': str(e)}), 400

# @app.route('/api/detection/calibrate',methods=['POST'])
# def calibrate_detection():
#     global pixels_per_mm
#     data=request.json; new_ppm=data.get('pixels_per_mm')
#     if new_ppm and new_ppm>0:
#         pixels_per_mm=float(new_ppm); return jsonify({'success':True,'pixels_per_mm':pixels_per_mm})
#     return jsonify({'success':False,'message':'Invalid value'}),400

# @app.route('/api/detection/update_grinder', methods=['POST'])
# def update_grinder_position():
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     try:
#         with camera_lock:
#             frame_to_process = last_frame.copy()
#         analyzer = SerratedBladeAnalyzer(frame_to_process)
#         analyzer.analyze_frame(use_stored_grinder=False)
#         if analyzer.grinder_tip:
#             set_current_grinder_tip(analyzer.grinder_tip)
#             tip = get_current_grinder_tip()
#             return jsonify({'success': True,
#                             'grinder_id': current_grinder_id,
#                             'grinder_tip': {'x': int(tip[0]), 'y': int(tip[1])},
#                             'message': f'Grinder {current_grinder_id} tip updated: {tip}'})
#         return jsonify({'success': False, 'message': 'Could not detect grinder'}), 404
#     except Exception as e:
#         return jsonify({'success': False, 'message': str(e)}), 500

# def _capture_grinder_average(duration_s=2.0, settle_ms=20):
#     """
#     Record grinder tip detections for `duration_s` and return a robust average.
#     The grinder vibrates laterally during operation, so a single-frame
#     'SET GRINDER' can land off-centre. Averaging ~50 frames and rejecting
#     gross V-fit failures (MAD on distance-from-median) gives a stable tip.
#     """
#     global last_frame
#     if not camera_active or last_frame is None:
#         return {'success': False, 'message': 'Camera not active'}

#     tips = []
#     deadline = time.time() + duration_s
#     while time.time() < deadline:
#         with camera_lock:
#             frame = last_frame.copy() if last_frame is not None else None
#         if frame is not None:
#             try:
#                 analyzer = SerratedBladeAnalyzer(frame)
#                 analyzer.preprocess_image()
#                 analyzer.detect_blade_and_grinder()
#                 if analyzer.grinder_tip is not None:
#                     tips.append((float(analyzer.grinder_tip[0]),
#                                  float(analyzer.grinder_tip[1])))
#             except Exception:
#                 pass
#         time.sleep(settle_ms / 1000.0)

#     if len(tips) < 3:
#         return {'success': False,
#                 'message': f'Too few valid detections ({len(tips)}) — check lighting/grinder'}

#     pts = np.array(tips, dtype=float)
#     med = np.median(pts, axis=0)
#     d   = np.linalg.norm(pts - med, axis=1)            # distance of each point from median
#     mad = np.median(np.abs(d - np.median(d))) or 1.0   # robust spread of those distances
#     keep = d <= (np.median(d) + 3.0 * 1.4826 * mad)    # 3σ-equivalent gate
#     kept = pts[keep] if keep.sum() >= 3 else pts

#     avg = kept.mean(axis=0)
#     tip = (int(round(avg[0])), int(round(avg[1])))
#     set_current_grinder_tip(tip)

#     return {
#         'success': True,
#         'tip': {'x': tip[0], 'y': tip[1]},
#         'samples':  int(len(tips)),
#         'kept':     int(keep.sum()),
#         'rejected': int(len(tips) - int(keep.sum())),
#         'x_std':    round(float(kept[:, 0].std()), 2),
#         'y_std':    round(float(kept[:, 1].std()), 2),
#         'grinder_id': current_grinder_id,
#     }

# @app.route('/api/detection/set_grinder_manual', methods=['POST'])
# def set_grinder_manual():
#     data = request.json or {}
#     try:
#         x = int(round(float(data['x'])))
#         y = int(round(float(data['y'])))
#     except (KeyError, TypeError, ValueError):
#         return jsonify({'success': False, 'message': 'Provide numeric x and y'}), 400
#     set_current_grinder_tip((x, y))
#     return jsonify({'success': True, 'grinder_id': current_grinder_id,
#                     'grinder_tip': {'x': x, 'y': y},
#                     'message': f'Grinder {current_grinder_id} tip set manually: ({x}, {y})'})


# @app.route('/api/detection/set_grinder_average', methods=['POST'])
# def set_grinder_average():
#     if not camera_active or last_frame is None:
#         return jsonify({'success': False, 'message': 'Camera not active'}), 400
#     data = request.json or {}
#     try:
#         duration = max(0.5, min(float(data.get('duration_s', 2.0)), 10.0))
#     except (TypeError, ValueError):
#         duration = 2.0
#     result = _capture_grinder_average(duration_s=duration)
#     return jsonify(result), (200 if result.get('success') else 404)

# @app.route('/api/detection/teeth_profiles', methods=['GET'])
# def get_teeth_profiles():
#     try:
#         frame_to_process = None
#         if camera_active and last_frame is not None:
#             with camera_lock: frame_to_process = last_frame.copy()
#         stored = get_current_grinder_tip()
#         if frame_to_process is None:
#             grinder = {'x': int(stored[0]), 'y': int(stored[1])} if stored else None
#             return jsonify({'success': True, 'teeth': [], 'grinder_tip': grinder, 'num_teeth': 0,
#                             'grinder_id': current_grinder_id})
#         analyzer = SerratedBladeAnalyzer(frame_to_process)
#         analyzer.preprocess_image(); analyzer.detect_blade_and_grinder()
#         analyzer.draw_grinder_v_overlay(frame_to_process)

#         profiles = analyzer.extract_tooth_profiles()
#         grinder_tip = None
#         if pitch_mm > 0 and stored:
#             global live_sine_overlay
#             _apx, _vly, _info, _ = apply_sine_correction(
#                 profiles, stored, pixels_per_mm, pitch_mm,
#                 blade_edge_points=analyzer.blade_edge_points,
#                 min_teeth_for_fit=2,
#                 baseline_poly_degree=1)
#             if _info:
#                 # Cache the fresh fit so the live feed (get_camera_frame) can draw it
#                 # when the dashboard overlay selector is in 'sine' mode.
#                 live_sine_overlay = {'apexes': _apx, 'valleys': _vly,
#                                      'info': _info, 't': time.time()}
#         if stored: analyzer.grinder_tip = stored
#         if analyzer.grinder_tip:
#             grinder_tip = {'x': int(analyzer.grinder_tip[0]), 'y': int(analyzer.grinder_tip[1])}
#         elif stored:
#             grinder_tip = {'x': int(stored[0]), 'y': int(stored[1])}
#         teeth = [{'tooth_id': t.tooth_id,
#                   'tip': {'x': int(t.grinding_point[0]), 'y': int(t.grinding_point[1])},
#                   'top_valley': {'x': int(t.top_valley[0]), 'y': int(t.top_valley[1])},
#                   'bottom_valley': {'x': int(t.bottom_valley[0]), 'y': int(t.bottom_valley[1])},
#                   'angle': round(float(t.angle), 2), 'height': round(float(t.height), 2)}
#                  for t in profiles]
#         return jsonify({'success': True, 'teeth': teeth, 'grinder_tip': grinder_tip,
#                         'num_teeth': len(teeth), 'grinder_id': current_grinder_id})
#     except Exception as e:
#         import traceback; traceback.print_exc()
#         return jsonify({'success': False, 'message': str(e)}), 500


# @app.route('/api/detection/grinder_status', methods=['GET'])
# def get_grinder_status():
#     tip = get_current_grinder_tip()
#     return jsonify({
#         'success': True,
#         'grinder_id': current_grinder_id,
#         'grinder_tip': ({'x': int(tip[0]), 'y': int(tip[1])} if tip else None),
#         'stored': tip is not None,
#         'all_tips': {str(k): {'x': int(v[0]), 'y': int(v[1])} for k, v in stored_grinder_tips.items()},
#         'mapping': CAMERA_SERIALS,
#     })


# if __name__=='__main__':
#     import logging
#     logging.getLogger('werkzeug').setLevel(logging.ERROR)   # stop logging every request

#     print("="*70)
#     print("🤖 BLADE GRINDER CONTROL SYSTEM")
#     print("   REG 134-136: Detection X/Y/Status")
#     print("   REG 137-140: START / GRINDER_READY / GRIND_START / E-STOP")
#     print("   REG 141:     TEETH_INSPECT  (Robot→PC, 1=record ON, 0=record OFF)")
#     print("   REG 142:     ROBOT_ANGLE    (Robot→PC, ×10 signed degrees)")
#     print("="*70)

#     # Warm up Spinnaker at startup so the slow one-time discovery happens
#     # BEFORE the operator starts working — camera start/switch is then fast.
#     try:
#         _get_pyspin_system()
#     except Exception as e:
#         print(f"⚠ Spinnaker warm-up failed (camera will retry on first start): {e}")

#     app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

"""
Standalone Flask Backend for Robot Control Dashboard
Blade grinder system – updated Modbus loop structure:

  PC writes X/Y + status  →  START=1
  Robot moves              →  sets GRINDER_READY=1
  PC detects (camera on)   →  writes new X/Y, sends GRIND_START=1
  Robot grinds, resets     →  GRIND_START=0  (ready for next cut)
  Repeat per tooth

  REG 141 = TEETH_INSPECT : Robot sets 1 to start calibration, 0 to stop.
  REG 142 = ROBOT_ANGLE   : Robot continuously writes current blade angle (×10, signed).
  While REG 141 = 1 the system records (angle, depth, pitch) only when
  REG 142 changes by ≥ 0.1° — no duplicate samples at the same angle.
  On falling edge (1→0) results are finalised; dashboard shows the
  angle and pitch at the maximum-depth point.

PERFORMANCE NOTES (Windows):
  * The PySpin System + CameraList are now initialised ONCE and kept alive for
    the whole process. Camera switches only DeInit/Init the camera itself —
    no more 40 s Spinnaker re-discovery on every switch.
  * debug_valid_mask.png write is disabled (DEBUG_SAVE_MASK flag).
  * werkzeug per-request logging silenced, debug=False, threaded=True.
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
import atexit
from dataclasses import dataclass
from typing import List, Tuple

from sine_fit_core import apply_sine_correction, draw_sine_overlay

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
# ONE CONTROLLER, MANY VIEWERS
# ------------------------------------------------------------------------------
# Exactly one browser ("the controller") may perform actions that change the
# machine (connect, config, start/stop, grind, camera, calibration, etc.).
# Everyone else is a read-only VIEWER: they see the live camera + status but
# their control actions are rejected with HTTP 423.
#
# EXCEPTION: EMERGENCY STOP (/api/estop) is allowed from ANY device, always.
#
# Enforcement is a single before_request hook keyed on the URL path, so no
# individual route needs to change. The controller proves identity with an
# opaque token (X-Control-Token header). A heartbeat keeps the lock alive;
# if the controller's tab closes or freezes, the lock auto-frees after
# CONTROL_TIMEOUT so the machine can't get stuck view-only forever.
# ══════════════════════════════════════════════════════════════════════════════
import uuid as _uuid
import hmac as _hmac
import os as _os
control_lock  = Lock()
control_state = {'token': None, 'name': None, 'last_heartbeat': 0.0}
CONTROL_TIMEOUT = 15.0   # seconds without a heartbeat before control auto-releases

# Control PIN — required to TAKE control (and to force a takeover). The current
# holder refreshing its own lock does NOT need it. Set it by creating a file
# named control_pin.txt next to this script; otherwise the default below is used.
CONTROL_PIN = '2468'
_pin_file = 'control_pin.txt'
if _os.path.exists(_pin_file):
    try:
        _p = open(_pin_file).read().strip()
        if _p:
            CONTROL_PIN = _p
            print(f"🔐 Control PIN loaded from {_pin_file}")
    except Exception as _e:
        print(f"⚠ Could not read {_pin_file}: {_e} — using default PIN")
else:
    print(f"🔐 Using DEFAULT control PIN. Create {_pin_file} with your own PIN to change it.")

def _check_pin(pin):
    # constant-time compare so the PIN can't be guessed by timing
    return _hmac.compare_digest(str(pin), str(CONTROL_PIN))

# POST paths that a viewer IS allowed to hit (everything else POST is gated)
CONTROL_EXEMPT_PATHS = {
    '/api/estop',              # EMERGENCY STOP — always allowed, any device
    '/api/control/acquire',
    '/api/control/release',
    '/api/control/heartbeat',
    '/api/control/status',
}

def _controller_active():
    """True if someone currently holds control and their heartbeat is fresh."""
    return (control_state['token'] is not None
            and (time.time() - control_state['last_heartbeat']) < CONTROL_TIMEOUT)

def _path_needs_control(path, method):
    # Only mutating actions are gated. GET/HEAD/OPTIONS (status, camera frame,
    # register reads, etc.) are always open so viewers can watch everything.
    if method != 'POST':
        return False
    if not path.startswith('/api/'):
        return False
    if path in CONTROL_EXEMPT_PATHS:
        return False
    return True

@app.before_request
def _enforce_single_controller():
    from flask import request
    if not _path_needs_control(request.path, request.method):
        return  # not a gated action → let it through
    token = request.headers.get('X-Control-Token', '')
    with control_lock:
        if not _controller_active():
            return jsonify({'success': False, 'control_error': True,
                            'message': 'No device has taken control yet. '
                                       'Click “Take Control” first.'}), 423
        if token != control_state['token']:
            return jsonify({'success': False, 'control_error': True,
                            'message': f"View-only mode — {control_state['name']} "
                                       f"currently has control."}), 423
    # token matches active controller → allowed to proceed

@app.route('/api/control/status', methods=['GET'])
def control_status():
    from flask import request
    token = request.headers.get('X-Control-Token', '') or request.args.get('token', '')
    with control_lock:
        active = _controller_active()
        return jsonify({
            'success': True,
            'has_controller': active,
            'controller_name': control_state['name'] if active else None,
            'you_are_controller': bool(active and token and token == control_state['token']),
            'timeout_s': CONTROL_TIMEOUT,
        })

@app.route('/api/control/acquire', methods=['POST'])
def control_acquire():
    from flask import request
    data  = request.json or {}
    name  = (str(data.get('name') or 'Operator')).strip()[:40] or 'Operator'
    force = bool(data.get('force', False))
    pin   = data.get('pin', '')
    hdr_token = request.headers.get('X-Control-Token', '')
    with control_lock:
        active = _controller_active()
        # Current holder refreshing its own lock (heartbeat / rename) → no PIN.
        if active and hdr_token and hdr_token == control_state['token']:
            control_state['last_heartbeat'] = time.time()
            if name:
                control_state['name'] = name
            return jsonify({'success': True, 'token': hdr_token,
                            'name': control_state['name'], 'reacquired': True})
        # Any real take-over of control requires the PIN.
        if not _check_pin(pin):
            return jsonify({'success': False, 'pin_error': True,
                            'message': 'Incorrect control PIN.'}), 403
        if active and not force:
            return jsonify({'success': False, 'control_error': True,
                            'held_by': control_state['name'],
                            'message': f"{control_state['name']} currently has control."}), 409
        # PIN OK and (free, expired, or forced) → grant a fresh token.
        new_token = _uuid.uuid4().hex
        control_state['token'] = new_token
        control_state['name']  = name
        control_state['last_heartbeat'] = time.time()
        print(f"🕹  Control acquired by '{name}'"
              f"{' (FORCED takeover)' if (active and force) else ''}")
        return jsonify({'success': True, 'token': new_token, 'name': name,
                        'forced': bool(active and force)})

@app.route('/api/control/heartbeat', methods=['POST'])
def control_heartbeat():
    from flask import request
    token = request.headers.get('X-Control-Token', '') or (request.json or {}).get('token', '')
    with control_lock:
        if _controller_active() and token == control_state['token']:
            control_state['last_heartbeat'] = time.time()
            return jsonify({'success': True, 'you_are_controller': True})
        active = _controller_active()
        return jsonify({'success': True, 'you_are_controller': False,
                        'has_controller': active,
                        'controller_name': control_state['name'] if active else None})

@app.route('/api/control/release', methods=['POST'])
def control_release():
    from flask import request
    # Accept token via header, ?token= (for navigator.sendBeacon), or JSON body.
    token = (request.headers.get('X-Control-Token', '')
             or request.args.get('token', '')
             or (request.get_json(silent=True) or {}).get('token', ''))
    with control_lock:
        if control_state['token'] is not None and token == control_state['token']:
            print(f"🕹  Control released by '{control_state['name']}'")
            control_state['token'] = None
            control_state['name']  = None
            control_state['last_heartbeat'] = 0.0
            return jsonify({'success': True, 'released': True})
        return jsonify({'success': True, 'released': False})

# ── Debug flags ───────────────────────────────────────────────────────────────
DEBUG_SAVE_MASK = False   # True → write debug_valid_mask.png on every analysis (SLOW on Windows)

# ── Grinder detection tuning ──────────────────────────────────────────────────
# Search area for the grinder's left edge. The scan starts to the RIGHT of the
# blade's actual right edge (+ GRINDER_GAP_FRAC) and covers rows from
# GRINDER_SCAN_TOP_FRAC*h down to GRINDER_SCAN_BOT_FRAC*h.
GRINDER_SCAN_TOP_FRAC = 1.0 / 3.0   # top of scan band (fraction of image height)
GRINDER_SCAN_BOT_FRAC = 1.0         # bottom of scan band
GRINDER_GAP_FRAC      = 0.02        # gap right of the blade edge (fraction of width)
GRINDER_MIN_DARK_PX   = 5           # min dark pixels in a row to count as grinder
BLADE_LIMIT_FRAC      = 0.45        # blade lives in the left N of the frame

# Threshold for "grinder dark". 'otsu' self-calibrates on the scan region and is
# strongly recommended: the grinder is near-black (~18 grey) while the background
# is ~150-190 WITH a top-to-bottom gradient, so any fixed fraction of a
# background mean sampled at the top can fall below the background at the bottom
# and make the scan return vignette contours instead of the grinder.
# Set to a number (e.g. 100) to force an absolute grey level instead.
GRINDER_THRESH_MODE = 'otsu'        # 'otsu' | <int absolute grey level>

# A real V-grinder has a known included angle. If the fitted V is outside this
# range the fit is not a grinder (glare, blade, empty frame) → report no tip
# rather than silently handing the robot a bogus coordinate.
GRINDER_ANGLE_MIN = 75.0
GRINDER_ANGLE_MAX = 105.0
# If the V-fit fails, fall back to the leftmost dark pixel (the ROUNDED NOSE,
# ~12 px / 0.16 mm right of the true vertex)? False = safer: report no tip.
GRINDER_ALLOW_NOSE_FALLBACK = False

# ── Teeth detection tuning (LATTICE method) ───────────────────────────────────
# The blade is water-jet cut, so the teeth form a REGULAR LATTICE: every tooth
# sits at y_k = phase + k*pitch. Instead of finding each tooth independently
# (which produces duplicates/misses and needs a dedupe hack), we fit those two
# numbers using every sample of the edge, then generate the teeth from the model.
#
# Why this survives damage: bending/breakage is LATERAL (X) and amplitude.
# The lattice constrains LONGITUDINAL (Y). They are orthogonal, so a bent or
# broken tooth cannot move the lattice — it becomes a measurable residual
# against the model instead of contaminating it.
DETECTION_METHOD = 'lattice'          # 'lattice' | 'legacy'

LATTICE_SUBPIX_EDGE     = True        # interpolate the grey ramp for the edge (13um -> ~3um)
LATTICE_DETREND_DEGREE  = 2           # polynomial for the blade's own curve/tilt
LATTICE_PITCH_SEARCH_LO = 0.6         # autocorrelation search bracket, x nominal pitch
LATTICE_PITCH_SEARCH_HI = 1.6         # (keep < 2.0 so it cannot lock onto the 2P harmonic)
LATTICE_SNAP_TOL_FRAC   = 0.20        # snap lattice->real peak within +/- this fraction of pitch
LATTICE_FREEZE_PITCH    = True        # measure pitch once per blade, then reuse (it cannot
                                      # legitimately change mid-blade on a water-jet blade)

# Confidence gates. Below any of these the frame REFUSES to produce coordinates
# (returns None -> status 2) rather than handing the robot a guess.
LATTICE_MIN_QUALITY        = 0.25     # autocorr peak / zero-lag: "is this periodic at all?"
LATTICE_MIN_PHASE_STRENGTH = 0.30     # phasor coherence: "do the teeth agree on a phase?"
LATTICE_MIN_SNAP_FRAC      = 0.50     # fraction of lattice teeth actually seen in the image
LATTICE_PITCH_TOLERANCE    = 0.15     # measured vs configured pitch mismatch -> reject
                                      # (catches "wrong blade loaded for this config")

# ── Damaged-tooth handling ────────────────────────────────────────────────────
# Classification thresholds, relative to the blade's own robust (median) geometry.
LATTICE_BENT_FRAC = 0.25   # |lateral residual| > this * tooth depth  -> BENT (also floored
                           # at 3*MAD of the blade's own residual scatter)
LATTICE_WORN_FRAC = 0.50   # tip amplitude     < this * median amp    -> WORN/BROKEN
# Policy:
#   True  = grind at the IDEAL lattice position (restores intended geometry;
#           a bent tooth gets more taken off its bent side). Recommended.
#   False = follow the observed (deformed) tooth positions, like the legacy code.
LATTICE_GRIND_IDEAL    = True
LATTICE_GRIND_MISSING  = True   # still grind the groove beside a missing tooth?
LATTICE_MAX_DAMAGED    = 0      # 0 = no limit; else refuse the blade above this many
                                # BENT+MISSING+WORN teeth (scrap threshold)

# ── Global state ──────────────────────────────────────────────────────────────
modbus_client    = None
client_connected = False

pyspin_system   = None   # kept alive for the whole process (see _get_pyspin_system)
pyspin_cam      = None
pyspin_cam_list = None   # kept alive for the whole process
camera_lock     = Lock()
camera_active   = False
camera_thread   = None
last_frame      = None
camera_config   = {'frame_rate': 30.0 }

blade_analyzer        = None
last_detection_result = None
last_sine_overlay     = None   # {'apexes','valleys','info'} from last successful sine fit (for live overlay)
last_overlay_pts      = None   # NoSine build: {'apexes','valleys'} deduped tops + candidate valleys for overlay
sine_overlay_enabled  = False  # toggled by the dashboard overlay selector ('sine' mode); hides the live envelope otherwise
live_sine_overlay     = None   # {'apexes','valleys','info','t'} fresh per-frame fit cached by /teeth_profiles for the live feed
detection_enabled     = False
pixels_per_mm           = 76.812
pitch_mm                = 0.0    # 0.0 = disabled; set from dashboard
grinder_positions_file  = 'grinder_positions.json'   # was grinder_position.json (singular)
stored_grinder_tips     = {}                         # {grinder_id: (x, y)}

# Camera-per-grinder mapping. Edit camera_serials.json after first run to assign
# FLIR serial numbers to each grinder. Empty string = fall back to index (gid-1).
camera_serials_file = 'camera_serials.json'
CAMERA_SERIALS      = {1: '', 2: ''}
current_grinder_id  = 1

# ── Inspection state ──────────────────────────────────────────────────────────
inspection_active       = False
inspection_thread       = None
last_inspection_summary = None
last_teeth_inspect_reg  = False
last_robot_angle        = 0.0

# ── Post-cut depth validation ─────────────────────────────────────────────────
last_grind_depth_mm     = None   # pre-cut groove depth (depth sent with last GRIND_START)
last_commanded_depth_mm = None   # commanded depth from last config (REG 131)
last_post_cut_result    = None   # last post-cut measurement (for dashboard + REG 150)




# ── Grinder position persistence ──────────────────────────────────────────────
def load_grinder_positions():
    """Load per-grinder tip positions. Migrates old single-file format if found."""
    global stored_grinder_tips
    import os, json
    if os.path.exists(grinder_positions_file):
        try:
            with open(grinder_positions_file, 'r') as f:
                data = json.load(f)
                stored_grinder_tips = {int(k): tuple(v) for k, v in data.items()}
                print(f"✓ Loaded grinder positions: {stored_grinder_tips}")
        except Exception as e:
            print(f"⚠ Could not load grinder positions: {e}")
            stored_grinder_tips = {}
    elif os.path.exists('grinder_position.json'):
        # Migrate old single-position file → grinder 1
        try:
            with open('grinder_position.json', 'r') as f:
                data = json.load(f)
                stored_grinder_tips = {1: tuple(data['grinder_tip'])}
                print(f"✓ Migrated old grinder position → grinder 1: {stored_grinder_tips[1]}")
                save_grinder_positions()
        except Exception as e:
            print(f"⚠ Migration failed: {e}")
            stored_grinder_tips = {}
    else:
        stored_grinder_tips = {}


def save_grinder_positions():
    import json
    try:
        data = {str(k): [int(v[0]), int(v[1])] for k, v in stored_grinder_tips.items()}
        with open(grinder_positions_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✓ Saved grinder positions: {stored_grinder_tips}")
    except Exception as e:
        print(f"⚠ Could not save grinder positions: {e}")


def get_current_grinder_tip():
    return stored_grinder_tips.get(current_grinder_id)


def set_current_grinder_tip(tip):
    stored_grinder_tips[current_grinder_id] = (int(tip[0]), int(tip[1]))
    save_grinder_positions()


def load_camera_serials():
    global CAMERA_SERIALS
    import os, json
    if os.path.exists(camera_serials_file):
        try:
            with open(camera_serials_file, 'r') as f:
                data = json.load(f)
                CAMERA_SERIALS = {int(k): str(v) for k, v in data.items()}
                print(f"✓ Loaded camera serials: {CAMERA_SERIALS}")
        except Exception as e:
            print(f"⚠ Could not load camera serials: {e}")
    else:
        save_camera_serials()  # write default file so user can edit it


def save_camera_serials():
    import json
    try:
        with open(camera_serials_file, 'w') as f:
            json.dump({str(k): v for k, v in CAMERA_SERIALS.items()}, f, indent=4)
    except Exception as e:
        print(f"⚠ Could not save camera serials: {e}")


load_grinder_positions()
load_camera_serials()


def _dedupe_clusters(profiles, y_threshold_mm=0.2, grinder_tip=None, select=None):
    """
    Collapse groups of teeth at nearly-identical Y positions into a single
    representative tooth per cluster. This removes duplicate detections
    without skipping any real teeth.

    Clustering rule: consecutive teeth within `y_threshold_mm` of each other
    are treated as the same physical tooth.

    Representative selection:
      - If grinder_tip provided → closest to grinder
      - Otherwise → rightmost (max X)  [original behavior]

    Args:
        profiles: list of ToothProfile, ordered by Y
        y_threshold_mm: cluster spacing threshold in millimeters
        grinder_tip: optional (x, y) to bias selection toward grinder

    Returns:
        New list with one representative tooth per cluster.
    """
    global pixels_per_mm

    if not profiles:
        return []

    threshold_px = y_threshold_mm * pixels_per_mm
    gt = grinder_tip if grinder_tip else get_current_grinder_tip()

    deduped = []
    i = 0
    while i < len(profiles):
        # Build cluster starting at i
        cluster = [profiles[i]]
        j = i + 1
        while j < len(profiles):
            if abs(cluster[-1].grinding_point[1] - profiles[j].grinding_point[1]) < threshold_px:
                cluster.append(profiles[j])
                j += 1
            else:
                break

        # Pick representative
        if len(cluster) == 1:
            best = cluster[0]
        elif select == 'tallest':
            # Most-protruding tip = the real tooth; a duplicate is a smaller bump beside it
            best = max(cluster, key=lambda t: t.height)
        elif gt:
            # Closest to grinder
            best = min(cluster, key=lambda t: (
                                                      (t.grinding_point[0] - gt[0]) ** 2 +
                                                      (t.grinding_point[1] - gt[1]) ** 2
                                              ) ** 0.5)
        else:
            # Rightmost (original behavior)
            best = max(cluster, key=lambda t: t.grinding_point[0])

        deduped.append(best)
        i = j  # jump past the entire cluster

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# LATTICE MATH — pitch, phase, sub-pixel refinement
# Pure functions (no camera/state) so they can be unit-tested.
# ══════════════════════════════════════════════════════════════════════════════

def _parabolic_offset(y0, y1, y2):
    """Sub-sample offset of a peak from the 3 samples straddling it (-1..1)."""
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))


def _subpix_edge_x(row_gray, x_int, level):
    """
    Refine an edge to sub-pixel by linear-interpolating where the grey level
    crosses `level` between pixel x_int and x_int+1.

    The thresholded pixel is quantised to 1 px (= 13 um at 76.8 px/mm) and moves
    with threshold/lighting drift; the intensity ramp does not.
    """
    n = len(row_gray)
    x0 = int(x_int)
    if x0 < 0 or x0 >= n - 1:
        return float(x0)
    v0 = float(row_gray[x0]); v1 = float(row_gray[x0 + 1])
    if v1 == v0:
        return float(x0)
    t = (level - v0) / (v1 - v0)
    if not (0.0 <= t <= 1.0):
        return float(x0)
    return x0 + t


def _coherence(y, w, tot, P):
    """Phasor coherence 0..1 of weighted samples at trial period P."""
    return float(abs(np.sum(w * np.exp(2j * np.pi * y / P))) / tot)


def _measure_pitch_px(y_coords, deviation, ppm, nominal_mm,
                      lo=LATTICE_PITCH_SEARCH_LO, hi=LATTICE_PITCH_SEARCH_HI,
                      coarse=256, fine=64):
    """
    Measure the tooth pitch by maximising PHASE COHERENCE over trial pitches:

        P* = argmax_P | SUM_i w_i * exp(2*pi*i*y_i/P) | / SUM_i w_i

    with w = positive part of the deviation (i.e. tooth tips vote).

    Why not autocorrelation: with only ~9 pitches in frame, a couple of broken
    teeth smear the autocorrelation peak and shift it badly — measured on a
    synthetic blade with 2 teeth broken off, autocorrelation put the pitch 8.8%
    low (140 px vs a true 153.6 px) and the frame got rejected. Coherence is
    sharp because a MISSING tooth simply casts no vote instead of distorting the
    estimate; the same case comes out +10.7 um (0.5%).

    Bracket note: sub-harmonics P/2, P/3... are also perfectly coherent, so `lo`
    must stay above 0.5. 2P is NOT coherent (alternate teeth cancel), so the
    upper bound is safe.

    Returns (pitch_px, coherence 0..1). Coherence ~0.7-0.8 on a good blade,
    near 0 when the edge isn't periodic.
    """
    y = np.asarray(y_coords, dtype=float)
    d = np.asarray(deviation, dtype=float)
    if y.size < 8:
        return float(nominal_mm * ppm), 0.0
    w = np.clip(d, 0.0, None)
    tot = float(w.sum())
    if tot <= 0:
        return float(nominal_mm * ppm), 0.0

    p_lo = max(2.0, lo * nominal_mm * ppm)
    p_hi = max(p_lo + 1.0, hi * nominal_mm * ppm)

    # Coarse scan for the main lobe, then a fine scan around it (keeps this at
    # ~300 evaluations instead of thousands — a few ms per frame).
    Ps = np.linspace(p_lo, p_hi, coarse)
    C = np.array([_coherence(y, w, tot, P) for P in Ps])
    k = int(np.argmax(C))
    step = (p_hi - p_lo) / (coarse - 1)
    f_lo = max(p_lo, Ps[k] - 2 * step)
    f_hi = min(p_hi, Ps[k] + 2 * step)
    Pf = np.linspace(f_lo, f_hi, fine)
    Cf = np.array([_coherence(y, w, tot, P) for P in Pf])
    j = int(np.argmax(Cf))
    P_best, C_best = float(Pf[j]), float(Cf[j])
    if 0 < j < len(Pf) - 1:      # parabolic refine on the fine grid
        off = _parabolic_offset(Cf[j - 1], Cf[j], Cf[j + 1])
        P_best += off * (Pf[1] - Pf[0])
    return float(P_best), max(0.0, min(1.0, C_best))


def _measure_phase(y_coords, deviation, P):
    """
    Given the pitch, find WHERE the lattice starts, using a phasor (circular
    mean) over every sample weighted by how tooth-like it is.

    Correct way to average a periodic quantity: no "first tooth" special case,
    no error accumulating along the blade.

    Returns (y0 in [0,P), strength 0..1) or (None, 0.0).
    """
    y = np.asarray(y_coords, dtype=float)
    w = np.clip(np.asarray(deviation, dtype=float), 0.0, None)   # tips are positive
    tot = w.sum()
    if tot <= 0 or P <= 0:
        return None, 0.0
    z = np.sum(w * np.exp(2j * np.pi * y / P))
    strength = float(abs(z) / tot)
    return float(((float(np.angle(z)) / (2.0 * np.pi)) * P) % P), max(0.0, min(1.0, strength))


# Pitch lock — a water-jet blade's pitch cannot change mid-blade, so measure it
# once and reuse. Keyed on the configured nominal, so changing blade/config
# re-measures automatically.
_pitch_lock = {'nominal_mm': None, 'pitch_px': None, 'quality': 0.0}

def _reset_pitch_lock(reason=''):
    _pitch_lock.update({'nominal_mm': None, 'pitch_px': None, 'quality': 0.0})
    if reason:
        print(f"🔓 Pitch lock reset ({reason})")


# ── Blade detection ───────────────────────────────────────────────────────────
@dataclass
class ToothProfile:
    tooth_id:        int
    apex_point:      Tuple[int, int]
    top_valley:      Tuple[int, int]
    bottom_valley:   Tuple[int, int]
    angle:           float
    grinding_point:  Tuple[int, int]
    height:          float
    move_to_grinder: Tuple[float, float]


class SerratedBladeAnalyzer:
    def __init__(self, image):
        self.image  = image
        self.gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape
        self.teeth_profiles     = []
        self.grinder_tip        = None
        self.blade_edge_points  = None
        self.grinder_edge_points = None
        self.grinder_upper_line = None
        self.grinder_lower_line = None
        self.grinder_edge_center = None
        self.grinder_roi = None
        self.grinder_thresh = None
        self.grad = 0.0
        self.m1 = 0
        self.m2 = 0

    def preprocess_image(self, blur_kernel=5):
        """
        Preprocess for grey-background images.

        Uses Otsu thresholding (auto-selects the blade/background boundary)
        instead of adaptive threshold which required near-white background.
        Morphological close removes small burr noise before edge extraction.
        """
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)

        # Otsu: automatically finds threshold between dark blade and grey background
        self._otsu_thresh, self.binary = cv2.threshold(
            self.blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological close: fills in burr gaps, smooths edge noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.binary = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        return self.binary

    def detect_blade_and_grinder(self, sampling_step=1):
        """
        Detect blade edge (left) and grinder tip (right) for grey-background images.

        Blade edge
        ----------
        Scans every row for the rightmost dark (blade) pixel in the left
        BLADE_LIMIT_FRAC of the image. Uses the Otsu binary from preprocess_image().

        Grinder edge
        ------------
        The grinder is a separate, near-black object to the right of the blade.
        For each row in the scan band we take the LEFTMOST dark pixel — that is
        the grinder's left edge. Those points feed _fit_grinder_v, which fits a
        line to each flank of the V and intersects them; the intersection (the
        virtual sharp vertex) is the tip, NOT the leftmost pixel.

        Thresholding (see GRINDER_THRESH_MODE)
        --------------------------------------
        Otsu is computed on the scan region itself. The previous approach —
        0.82 x (mean of a background patch sampled at the TOP of the frame) —
        is unsafe: measured on a real frame the background runs ~191 grey at the
        top but ~157 at the bottom (vignetting), so the threshold (157) landed
        BELOW the background it was applied to and every "grinder" point was
        actually background. Otsu on the region self-calibrates and sits in the
        wide gap between grinder (~18) and background (~150+).
        """
        h, w = self.gray.shape
        blade_limit = int(w * BLADE_LIMIT_FRAC)

        # ── Blade edge ────────────────────────────────────────────────────────────
        blade_edge = []
        for y in range(0, h, sampling_step):
            row = self.binary[y, :blade_limit]
            dark = np.where(row > 0)[0]
            if len(dark) > 10:  # skip rows with barely any blade
                blade_edge.append((int(dark.max()), y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None

        # ── Grinder scan area ─────────────────────────────────────────────────────
        # Anchor the left boundary to where the blade REALLY ends in this frame
        # (+ a gap), so the scan follows the grinder wherever it advances while
        # keeping the blade teeth out of it.
        if self.blade_edge_points is not None and len(self.blade_edge_points):
            blade_right = int(self.blade_edge_points[:, 0].max())
        else:
            blade_right = blade_limit
        gap_px  = max(8, int(w * GRINDER_GAP_FRAC))
        scan_x0 = min(blade_right + gap_px, w - 1)
        scan_y0 = max(0, int(h * GRINDER_SCAN_TOP_FRAC))
        scan_y1 = min(h, int(h * GRINDER_SCAN_BOT_FRAC))
        self.grinder_roi = (scan_x0, scan_y0, w - 1, scan_y1)   # for overlay/debug

        # ── Grinder threshold ─────────────────────────────────────────────────────
        region = self.gray[scan_y0:scan_y1, scan_x0:]
        if GRINDER_THRESH_MODE == 'otsu' and region.size:
            _t, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            grinder_thresh = int(_t)
        elif isinstance(GRINDER_THRESH_MODE, (int, float)):
            grinder_thresh = int(GRINDER_THRESH_MODE)
        else:
            grinder_thresh = 100
        self.grinder_thresh = grinder_thresh

        # ── Grinder edge (leftmost dark pixel per row) ────────────────────────────
        grinder_points = []
        for y in range(scan_y0, scan_y1, sampling_step):
            row = self.gray[y, scan_x0:]
            dark = np.where(row < grinder_thresh)[0]
            if len(dark) > GRINDER_MIN_DARK_PX:
                grinder_points.append((int(dark.min()) + scan_x0, y))

        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        # ── V-fit: tip = intersection of the two fitted flank lines ───────────────
        self.grinder_tip = None
        self.grinder_edge_center = None
        self.grinder_upper_line = None
        self.grinder_lower_line = None

        if self.grinder_edge_points is not None and len(self.grinder_edge_points) >= 6:
            tip, upper, lower, self.grad, self.m1, self.m2 = \
                self._fit_grinder_v(self.grinder_edge_points)

            # Reject an implausible V (glare / blade / empty frame). Otsu always
            # returns *some* threshold, so this is what stops a bogus tip from
            # reaching the robot.
            if tip is not None and not (GRINDER_ANGLE_MIN <= self.grad <= GRINDER_ANGLE_MAX):
                print(f"⚠ Grinder V rejected: included angle {self.grad:.1f}° outside "
                      f"{GRINDER_ANGLE_MIN}-{GRINDER_ANGLE_MAX}°")
                tip = None

            if tip is not None:
                self.grinder_tip = tip                 # the virtual sharp vertex
                self.grinder_edge_center = tip
                self.grinder_upper_line = upper
                self.grinder_lower_line = lower
            elif GRINDER_ALLOW_NOSE_FALLBACK:
                # Rounded nose — ~12 px (0.16 mm) right of the true vertex. Off by
                # default: a wrong tip is worse than no tip on a live machine.
                min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
                self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
                min_x = self.grinder_edge_points[min_x_idx, 0]
                tip_pts = self.grinder_edge_points[
                    np.abs(self.grinder_edge_points[:, 0] - min_x) < 10]
                self.grinder_edge_center = (
                    int(np.mean(tip_pts[:, 0])), int(np.mean(tip_pts[:, 1])))
                print("⚠ Grinder V-fit failed — using leftmost-pixel fallback (less accurate)")

        return self.blade_edge_points, self.grinder_tip

    def _generate_coordinates_sine(self, grinder_tip):
        global pixels_per_mm, pitch_mm

        gt_y = grinder_tip[1]
        gt_x = grinder_tip[0]

        corrected_apexes, corrected_valleys, info, above_profiles = apply_sine_correction(
            self.teeth_profiles,
            grinder_tip,
            pixels_per_mm,
            pitch_mm,
            blade_edge_points=self.blade_edge_points,
            min_teeth_for_fit=2,
            baseline_poly_degree=1,
        )

        if corrected_apexes is None or corrected_valleys is None or len(corrected_apexes) < 2:
            return None

        self._sine_corrected_apexes = corrected_apexes
        self._sine_corrected_valleys = corrected_valleys
        self._sine_info = info

        global last_sine_overlay
        last_sine_overlay = {'apexes': corrected_apexes,
                             'valleys': corrected_valleys, 'info': info}

        sorted_valleys = sorted(corrected_valleys, key=lambda v: v[1])
        closest_valley = None
        min_distance = float('inf')

        for (vx, vy) in sorted_valleys:
            move_x_mm = (gt_x - vx) / pixels_per_mm
            move_y_mm = (gt_y - vy) / pixels_per_mm
            if move_y_mm < 0.5:
                continue
            dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            if dist < min_distance:
                min_distance = dist
                depth_x_mm = abs(info['A_mean']) / pixels_per_mm
                apexes_above = [a for a in corrected_apexes if a[1] < vy]
                ct_lbl = len(apexes_above)
                closest_valley = {
                    'valley_x': vx,
                    'valley_y': vy,
                    'move_x_mm': move_x_mm,
                    'move_y_mm': move_y_mm,
                    'depth_x_mm': depth_x_mm,
                    'between_teeth': f"{ct_lbl}-{ct_lbl + 1}",
                    'distance_mm': dist,
                }

        if not closest_valley:
            return None

        num_teeth = len([a for a in corrected_apexes if a[1] < gt_y])

        return {
            'valley_id': closest_valley['between_teeth'],
            'x_mm': round(float(closest_valley['move_y_mm']), 2),
            'y_mm': round(float(closest_valley['move_x_mm']), 2),
            'depth_x_mm': round(float(closest_valley['depth_x_mm']), 2),
            'valley_x_px': int(closest_valley['valley_x']),
            'valley_y_px': int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(grinder_tip[0]),
            'grinder_tip_y_px': int(grinder_tip[1]),
            'num_teeth': num_teeth,
            'distance_mm': round(float(closest_valley['distance_mm']), 2),
            'status': 1,
            'all_valleys': [],
            'sine_fit': True,
            'pitch_mm': pitch_mm,
        }
    def _fit_grinder_v(self, points, min_points_per_side=3, max_iter=3, trim_frac=0.15):
        """
        Fit two lines (upper + lower halves of V) to grinder edge points and return
        the intersection (V-apex = grinder tip).

        Iterative refinement:
          1. Find point of minimum X → rough apex estimate
          2. Split points into above/below apex by Y
          3. Fit a line to each half using least squares on x = m*y + b
             (swapped because the V is near-vertical; avoids infinite slopes)
          4. Trim outliers (worst `trim_frac` residuals) and refit
          5. Solve intersection of the two lines for refined apex
          6. Repeat using refined apex as new split point

        Args:
            points: Nx2 array of (x, y) grinder edge points
            min_points_per_side: need at least this many points per half
            max_iter: refinement iterations (2–3 is enough)
            trim_frac: fraction of worst-fit points to discard per iteration

        Returns:
            (tip, upper_line_params, lower_line_params) or (None, None, None) on failure
            - tip = (x, y) integer tuple (V-apex)
            - *_line_params = (m, b) for x = m*y + b
        """
        if len(points) < 2 * min_points_per_side:
            return None, None, None, 0.0, 0.0, 0.0

        pts = np.asarray(points, dtype=float)
        xs, ys = pts[:, 0], pts[:, 1]

        # ── Initial split: use argmin(x) as rough apex ────────────────────────────
        split_y = ys[np.argmin(xs)]

        upper_line = None
        lower_line = None
        tip = None

        for _ in range(max_iter):
            upper_mask = ys < split_y
            lower_mask = ys > split_y

            if upper_mask.sum() < min_points_per_side or lower_mask.sum() < min_points_per_side:
                break

            upper_line = self._fit_line_trimmed(xs[upper_mask], ys[upper_mask], trim_frac)
            lower_line = self._fit_line_trimmed(xs[lower_mask], ys[lower_mask], trim_frac)

            if upper_line is None or lower_line is None:
                break

            # Solve intersection: x = m1*y + b1 = m2*y + b2
            # → y = (b1 - b2) / (m2 - m1), then x = m1*y + b1
            m1, b1 = upper_line
            m2, b2 = lower_line
            denom = (m2 - m1)
            if abs(denom) < 1e-6:
                # Lines are parallel — can't find intersection
                break

            y_apex = (b1 - b2) / denom
            x_apex = m1 * y_apex + b1
            tip = (int(round(x_apex)), int(round(y_apex)))

            # Use new apex as split point for next iteration
            new_split = y_apex
            if abs(new_split - split_y) < 0.5:
                break  # converged
            split_y = new_split

        if tip is None or upper_line is None or lower_line is None:
            return None, None, None, 0.0, 0.0, 0.0

        # Sanity check: the fitted tip should lie near the observed points
        # (within the y-range and not wildly off in x)
        y_min, y_max = ys.min(), ys.max()
        if not (y_min - 20 <= tip[1] <= y_max + 20):
            return None, None, None, 0.0, 0.0, 0.0

        x_min = xs.min()
        if tip[0] < x_min - 30 or tip[0] > xs.max() + 30:
            return None, None, None, 0.0, 0.0, 0.0

        import math
        gradient = math.degrees(math.atan(abs((m1-m2)/(1+m1*m2))))
        return tip, upper_line, lower_line, gradient, m1 , m2

    @staticmethod
    def _fit_line_trimmed(xs, ys, trim_frac=0.15):
        """
        Fit x = m*y + b using least squares, then discard the worst `trim_frac`
        of points by residual and refit once. Returns (m, b) or None.

        We fit x as a function of y (not y = f(x)) because the V-shape edges
        are near-vertical in the image; x = m*y + b handles vertical lines
        gracefully while y = m*x + b would blow up.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if len(xs) < 2:
            return None

        # First pass
        try:
            m, b = np.polyfit(ys, xs, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        if len(xs) < 4:
            return (float(m), float(b))

        # Trim worst residuals and refit
        residuals = np.abs(xs - (m * ys + b))
        threshold = np.quantile(residuals, 1.0 - trim_frac)
        keep = residuals <= threshold

        if keep.sum() < 2:
            return (float(m), float(b))

        try:
            m2, b2 = np.polyfit(ys[keep], xs[keep], 1)
            return (float(m2), float(b2))
        except (np.linalg.LinAlgError, ValueError):
            return (float(m), float(b))

    def extract_tooth_profiles(self, window_size=10):
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return []
        x_coords = self.blade_edge_points[:, 0]
        y_coords = self.blade_edge_points[:, 1]
        x_smooth = ndimage.gaussian_filter1d(x_coords, sigma=3)

        valid_mask = np.ones(len(y_coords), dtype=bool)
        if self.image is not None:
            img_array = np.array(self.image) if not isinstance(self.image, np.ndarray) else self.image
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if img_array.ndim == 3 else img_array
            for i, y in enumerate(y_coords.astype(int)):
                if 0 <= y < gray.shape[0] and np.all(gray[y, :] == 255):
                    valid_mask[i] = False

        n = len(x_smooth)
        baseline_half = min(max(window_size * 4, n // 6), n // 3)
        pad    = baseline_half
        padded = np.pad(x_smooth, pad, mode='reflect')
        baseline = np.array([np.median(padded[max(0, i): i + 2 * pad + 1]) for i in range(n)])
        if not valid_mask.all():
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0:
                for i in np.where(~valid_mask)[0]:
                    baseline[i] = baseline[valid_idx[np.argmin(np.abs(valid_idx - i))]]

        deviation  = x_smooth - baseline
        dev_valid  = deviation[valid_mask] if valid_mask.any() else deviation
        prominence = max(2.0, (np.percentile(dev_valid, 75) - np.percentile(dev_valid, 25)) * 0.20) if len(dev_valid) > 4 else 2.0

        peaks, valleys = [], []
        for i in range(window_size, n - window_size):
            if not valid_mask[i]: continue
            window_dev = deviation[i - window_size: i + window_size]
            if deviation[i] == np.max(window_dev) and deviation[i] > prominence:
                peaks.append(i)
            elif deviation[i] == np.min(window_dev) and deviation[i] < -prominence:
                valleys.append(i)
        peaks   = self._filter_close_points(peaks,   window_size)
        valleys = self._filter_close_points(valleys, window_size)

        # Debug visualisation — DISABLED by default. Writing a PNG on every
        # analysis call (overlay poll = every 300 ms) is a major slowdown,
        # especially on Windows where Defender scans each file write.
        if DEBUG_SAVE_MASK and self.image is not None:
            mask_img = img_array.copy()
            for i, y in enumerate(y_coords.astype(int)):
                if 0 <= y < mask_img.shape[0]:
                    x = int(x_coords[i])
                    if 0 <= x < mask_img.shape[1]:
                        cv2.circle(mask_img, (x, y), 3, (0,255,0) if valid_mask[i] else (0,0,255), -1)
            for i, y in enumerate(y_coords.astype(int)):
                bx = int(baseline[i])
                if 0 <= y < mask_img.shape[0] and 0 <= bx < mask_img.shape[1]:
                    cv2.circle(mask_img, (bx, y), 2, (0,165,255), -1)
            cv2.imwrite("debug_valid_mask.png", mask_img)

        tooth_profiles = []
        for tooth_id, peak_idx in enumerate(peaks, start=1):
            valleys_above = [v for v in valleys if v < peak_idx]
            valleys_below = [v for v in valleys if v > peak_idx]
            if len(valleys_above) == 0 and len(valleys_below) > 0:
                s  = [x_smooth[idx] for idx in range(0, min(window_size//2, peak_idx))]
                sy = [y_coords[idx] for idx in range(0, min(window_size//2, peak_idx))]
                top_valley    = (int(np.mean(s)),int(np.mean(sy))) if s else (int(x_smooth[0]),int(y_coords[0]))
                bottom_valley = (int(x_smooth[valleys_below[0]]),int(y_coords[valleys_below[0]]))
            elif len(valleys_below) == 0 and len(valleys_above) > 0:
                start = peak_idx; end = min(len(x_smooth)-1, peak_idx+window_size*2)
                s  = [x_smooth[idx] for idx in range(start,end)]
                sy = [y_coords[idx] for idx in range(start,end)]
                top_valley    = (int(x_smooth[valleys_above[-1]]),int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(np.mean(s)),int(np.mean(sy))) if s else (int(x_smooth[end]),int(y_coords[end]))
            elif len(valleys_above) > 0 and len(valleys_below) > 0:
                top_valley    = (int(x_smooth[valleys_above[-1]]),int(y_coords[valleys_above[-1]]))
                bottom_valley = (int(x_smooth[valleys_below[0]]), int(y_coords[valleys_below[0]]))
            else:
                continue
            tooth_point = (int(x_smooth[peak_idx]),int(y_coords[peak_idx]))
            height      = abs(tooth_point[0] - ((top_valley[0]+bottom_valley[0])/2))
            angle       = self._calculate_tooth_angle(top_valley,tooth_point,bottom_valley)
            move_to_grinder = ((self.grinder_tip[0]-tooth_point[0],self.grinder_tip[1]-tooth_point[1]) if self.grinder_tip else (0,0))
            tooth_profiles.append(ToothProfile(
                tooth_id=tooth_id,apex_point=tooth_point,
                top_valley=top_valley,bottom_valley=bottom_valley,
                angle=angle,grinding_point=tooth_point,
                height=height,move_to_grinder=move_to_grinder))
        return tooth_profiles

    def draw_grinder_v_overlay(self, img):
        """Draw the fitted V (two lines + apex point) on a BGR image in-place."""
        if self.grinder_upper_line is None or self.grinder_lower_line is None:
            return img
        if self.grinder_edge_points is None or len(self.grinder_edge_points) == 0:
            return img

        h, w = img.shape[:2]
        ys = self.grinder_edge_points[:, 1]
        y_min, y_max = int(ys.min()), int(ys.max())

        # Upper line (x = m*y + b) — draw in cyan
        m1, b1 = self.grinder_upper_line
        if self.grinder_tip:
            y_end = self.grinder_tip[1]
        else:
            y_end = (y_min + y_max) // 2
        pt1 = (int(round(m1 * y_min + b1)), y_min)
        pt2 = (int(round(m1 * y_end + b1)), int(y_end))
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

        # Lower line — draw in magenta
        m2, b2 = self.grinder_lower_line
        if self.grinder_tip:
            y_start = self.grinder_tip[1]
        else:
            y_start = (y_min + y_max) // 2
        pt3 = (int(round(m2 * y_start + b2)), int(y_start))
        pt4 = (int(round(m2 * y_max + b2)), y_max)
        cv2.line(img, pt3, pt4, (255, 0, 255), 2)

        # Apex (V-tip) — big yellow circle with crosshair
        if self.grinder_tip:
            tx, ty = int(self.grinder_tip[0]), int(self.grinder_tip[1])
            cv2.circle(img, (tx, ty), 10, (0, 255, 255), 2)
            cv2.line(img, (tx - 15, ty), (tx + 15, ty), (0, 255, 255), 1)
            cv2.line(img, (tx, ty - 15), (tx, ty + 15), (0, 255, 255), 1)
            cv2.putText(img, "V-APEX ", (tx + 12, ty - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img, f"{float(self.grad)}", (tx + 100, ty - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Raw edge points — small green dots
        for x, y in self.grinder_edge_points:
            cv2.circle(img, (int(x), int(y)), 1, (0, 200, 0), -1)

        return img

    def _filter_close_points(self, points, min_distance):
        if not points: return []
        filtered = [points[0]]
        for p in points[1:]:
            if p - filtered[-1] >= min_distance: filtered.append(p)
        return filtered

    def _calculate_tooth_angle(self, top_valley, tooth_point, bottom_valley):
        try:
            v1 = np.array(tooth_point) - np.array(top_valley)
            v2 = np.array(bottom_valley) - np.array(tooth_point)
            return float(np.degrees(np.arctan2(v2[1],v2[0]) - np.arctan2(v1[1],v1[0])))
        except: return 0.0



    def _groove_sharpness_score(self, ct_tip, nt_tip, perp_half=16):
        """
        Measure edge contrast in the V-groove by sampling perpendicular to
        the blade edge at every detected edge pixel within the groove y-range.

        At each edge point (x, y):
          dark side  = gray[y, x - perp_half]  (a few px into the blade)
          bright side = gray[y, x + perp_half]  (a few px into the background)
          contrast   = bright - dark

        Mean contrast = sharpness score.  Higher = sharper.
        Immune to burrs because it measures intensity difference across the
        edge, not the geometry of the edge itself.

        perp_half : pixels to step each side of the edge (default 8).
                    Increase to average over a wider sampling band.
        """
        if self.blade_edge_points is None or len(self.blade_edge_points) == 0:
            return None

        h, w = self.gray.shape

        # Filter edge points to the groove y-range (between the two tooth tips)
        y_min = min(ct_tip[1], nt_tip[1])
        y_max = max(ct_tip[1], nt_tip[1])
        ys   = self.blade_edge_points[:, 1]
        mask = (ys >= y_min) & (ys <= y_max)
        groove_pts = self.blade_edge_points[mask]

        if len(groove_pts) == 0:
            return None

        contrasts = []
        for x, y in groove_pts:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < h):
                continue
            x_dark   = max(0,     xi - perp_half)  # into blade (dark side)
            x_bright = min(w - 1, xi + perp_half)  # into background (bright side)
            contrast = float(self.gray[yi, x_bright]) - float(self.gray[yi, x_dark])
            if contrast > 0:   # guard against inverted or invalid samples
                contrasts.append(contrast)

        return float(np.mean(contrasts)) if contrasts else None

    def _get_closest_pair_stats(self, sharp_threshold=150.0):
        """
        Uses cluster-deduped tooth list, then iterates all consecutive pairs.
        Returns pitch, depth, sharpness for the pair closest to grinder.
        """
        global pixels_per_mm

        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        if not grinder_tip or len(self.teeth_profiles) < 2:
            return None

        # Dedupe clusters first
        deduped = _dedupe_clusters(
            self.teeth_profiles,
            y_threshold_mm=1,
            grinder_tip=grinder_tip,
        )

        if len(deduped) < 2:
            return None

        closest = None
        min_dist = float("inf")

        for i in range(len(deduped) - 1):
            ct = deduped[i]
            nt = deduped[i + 1]

            valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
            valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
            dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest = {
                    "ct_id": ct.tooth_id,
                    "nt_id": nt.tooth_id,
                    "pitch_mm": abs(ct.grinding_point[1] - nt.grinding_point[1]) / pixels_per_mm,
                    "depth_mm": abs(ct.grinding_point[0] - ct.bottom_valley[0]) / pixels_per_mm,
                    "_ct_tip": ct.grinding_point,
                    "_nt_tip": nt.grinding_point,
                }

        if not closest:
            return None

        sharpness = self._groove_sharpness_score(
            closest.pop("_ct_tip"),
            closest.pop("_nt_tip"),
        )
        closest["sharpness_score"] = round(sharpness, 2) if sharpness is not None else None
        closest["is_sharp"] = (sharpness is not None and sharpness >= sharp_threshold)
        return closest

    def measure_grinder_valley_depth(self):
        """
        Measure the groove depth of the valley DIRECTLY ACROSS from the grinder
        tip — the tooth that was just cut. Depth is the SAME calculation as
        detection depth: average X of the two neighbouring tooth tips minus the
        valley bottom X. Only this one valley — no averaging across the blade.
        Returns dict or None.
        """
        global pixels_per_mm
        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        if not grinder_tip or len(self.teeth_profiles) < 2:
            return None
        deduped = _dedupe_clusters(self.teeth_profiles, y_threshold_mm=1.0,
                                   grinder_tip=grinder_tip)
        if len(deduped) < 2:
            return None
        gt_y = grinder_tip[1]
        best, best_d = None, float('inf')
        for i in range(len(deduped) - 1):
            ct, nt = deduped[i], deduped[i + 1]
            valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2.0
            d = abs(valley_y - gt_y)
            if d < best_d:
                best_d, best = d, (ct, nt)
        if best is None:
            return None
        ct, nt = best
        avg_tip_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2.0
        valley_x  = ct.bottom_valley[0]
        depth_mm  = abs(avg_tip_x - valley_x) / pixels_per_mm
        valley_y  = (ct.grinding_point[1] + nt.grinding_point[1]) / 2.0
        return {
            'depth_mm':          round(float(depth_mm), 3),
            'valley_id':         f'{ct.tooth_id}-{nt.tooth_id}',
            'valley_x_px':       int(valley_x),
            'valley_y_px':       int(valley_y),
            'grinder_offset_px': int(best_d),
        }

    def _edge_signal(self):
        """
        Build the 1-D edge signal used by the lattice method.

        Returns dict(y, x, dev, baseline_fn) or None:
          y   : row coordinate along the blade
          x   : lateral edge position (sub-pixel if LATTICE_SUBPIX_EDGE)
          dev : x - baseline  (the teeth as a wave centred on zero)
          baseline_fn(y) : the blade's own curve/tilt

        The baseline is a low-order polynomial, NOT a rolling median: the blade
        edge is essentially straight, the teeth average out over the fit, and a
        median window only a few pitches wide can absorb the very periodicity
        we're trying to measure (and its width changed with how much blade was
        in frame).
        """
        if self.blade_edge_points is None or len(self.blade_edge_points) < 20:
            return None
        pts = np.asarray(self.blade_edge_points, dtype=float)
        y = pts[:, 1].copy()
        x = pts[:, 0].copy()

        if LATTICE_SUBPIX_EDGE:
            level = float(getattr(self, '_otsu_thresh', 0) or 0)
            if level > 0:
                h, w = self.gray.shape
                for i in range(len(x)):
                    yi = int(y[i])
                    if 0 <= yi < h:
                        x[i] = _subpix_edge_x(self.gray[yi, :], int(x[i]), level)

        # Mild smoothing (sigma=3 px out of a ~154 px pitch — harmless)
        x_s = ndimage.gaussian_filter1d(x, sigma=3)

        deg = max(1, int(LATTICE_DETREND_DEGREE))
        try:
            coeffs = np.polyfit(y, x_s, deg)
        except (np.linalg.LinAlgError, ValueError):
            return None
        baseline_fn = np.poly1d(coeffs)
        dev = x_s - baseline_fn(y)
        return {'y': y, 'x': x_s, 'dev': dev, 'baseline': baseline_fn}

    def _generate_coordinates_lattice(self):
        """
        LATTICE method — see the DETECTION_METHOD config block.

        1. Build the edge signal and detrend it.
        2. Measure the pitch P from the whole signal (autocorrelation).
        3. Measure the phase (phasor over all samples)  ->  teeth at y0 + k*P.
        4. Snap each lattice tooth to the nearest real peak (keep the prediction
           if there isn't one — that's a MISSING/broken tooth, not a gap).
        5. Classify each tooth OK / BENT / WORN / MISSING against the blade's own
           robust (median) geometry.
        6. Valleys = lattice + P/2, X from the baseline + median valley depth.
        7. Pick the valley nearest the grinder; refuse the frame if confidence
           is low or the measured pitch disagrees with the configured blade.
        """
        global pixels_per_mm, pitch_mm, last_overlay_pts

        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        if not grinder_tip:
            return None
        gt_x, gt_y = float(grinder_tip[0]), float(grinder_tip[1])

        sig = self._edge_signal()
        if sig is None:
            return None
        y, dev, baseline = sig['y'], sig['dev'], sig['baseline']
        x_s = sig['x']

        nominal_mm = pitch_mm if pitch_mm > 0 else 2.0

        # ── 2. Pitch (measured, with optional lock) ───────────────────────────
        if (LATTICE_FREEZE_PITCH and _pitch_lock['pitch_px']
                and _pitch_lock['nominal_mm'] == nominal_mm):
            P = float(_pitch_lock['pitch_px'])
            quality = float(_pitch_lock['quality'])
        else:
            P, quality = _measure_pitch_px(y, dev, pixels_per_mm, nominal_mm)
            if quality >= LATTICE_MIN_QUALITY and LATTICE_FREEZE_PITCH:
                _pitch_lock.update({'nominal_mm': nominal_mm, 'pitch_px': P,
                                    'quality': quality})
                print(f"🔒 Pitch locked: {P:.2f} px = {P/pixels_per_mm:.3f} mm "
                      f"(configured {nominal_mm:.2f} mm, quality {quality:.2f})")

        if quality < LATTICE_MIN_QUALITY:
            print(f"⚠ Lattice: edge is not periodic (quality {quality:.2f} < "
                  f"{LATTICE_MIN_QUALITY}) — no teeth?")
            return None

        # Cross-check the blade against the configuration.
        ratio = (P / pixels_per_mm) / nominal_mm
        if not (1.0 - LATTICE_PITCH_TOLERANCE <= ratio <= 1.0 + LATTICE_PITCH_TOLERANCE):
            print(f"⚠ Pitch mismatch: configured {nominal_mm:.2f} mm but measured "
                  f"{P/pixels_per_mm:.2f} mm — wrong blade or wrong config?")
            return None

        # ── 3. Phase → the lattice ────────────────────────────────────────────
        y0, strength = _measure_phase(y, dev, P)
        if y0 is None or strength < LATTICE_MIN_PHASE_STRENGTH:
            print(f"⚠ Lattice: weak phase coherence ({strength:.2f})")
            return None

        y_min, y_max = float(y.min()), float(y.max())
        k0 = int(np.floor((y_min - y0) / P))
        k1 = int(np.ceil((y_max - y0) / P))
        lattice_ys = np.array([y0 + k * P for k in range(k0, k1 + 1)], dtype=float)
        lattice_ys = lattice_ys[(lattice_ys >= y_min) & (lattice_ys <= y_max)]
        if lattice_ys.size < 2:
            return None

        # ── 4. Snap to the real image (trust the model, verify with pixels) ───
        tol = LATTICE_SNAP_TOL_FRAC * P
        teeth = []          # dicts: ty, tx, amp, observed
        for ty_pred in lattice_ys:
            m = np.abs(y - ty_pred) <= tol
            if not m.any():
                teeth.append({'ty': float(ty_pred),
                              'tx': float(baseline(ty_pred)),
                              'amp': 0.0, 'observed': False})
                continue
            idx = np.where(m)[0]
            j = int(idx[np.argmax(dev[idx])])
            off = 0.0
            if 0 < j < len(dev) - 1:
                off = _parabolic_offset(dev[j - 1], dev[j], dev[j + 1])
            teeth.append({'ty': float(y[j] + off), 'tx': float(x_s[j]),
                          'amp': float(dev[j]), 'observed': True})

        snapped = [t for t in teeth if t['observed']]
        snap_frac = len(snapped) / float(len(teeth))
        if snap_frac < LATTICE_MIN_SNAP_FRAC:
            print(f"⚠ Lattice: only {snap_frac*100:.0f}% of teeth visible")
            return None

        # ── 5. Robust geometry + damage classification ────────────────────────
        # Medians, not means: one broken tooth must not set the cut depth.
        amps = np.array([t['amp'] for t in snapped], dtype=float)
        A_tip = float(np.median(amps))

        # Valley depth from the signal at the half-offset positions.
        val_devs = []
        for t in teeth:
            vy = t['ty'] + 0.5 * P
            m = np.abs(y - vy) <= tol
            if m.any():
                val_devs.append(float(np.min(dev[np.where(m)[0]])))
        A_val = float(np.median(val_devs)) if val_devs else -A_tip
        depth_px = max(1e-6, A_tip - A_val)          # peak-to-peak tooth depth
        depth_mm = depth_px / pixels_per_mm

        bent_lim = LATTICE_BENT_FRAC * depth_px
        worn_lim = LATTICE_WORN_FRAC * A_tip

        # Teeth at the very ends of the signal are partial/unreliable (the snap
        # window runs off the end and the detrend is least constrained there),
        # so classify them as EDGE rather than inventing damage.
        edge_guard = 0.5 * P

        # Adaptive floor: measure the blade's own residual noise (MAD) and don't
        # call anything BENT that's within it. Without this, ~0.1 mm of ordinary
        # residual scatter trips a fixed threshold and every blade looks bent.
        _res = []
        for t in teeth:
            if t['observed'] and (y_min + edge_guard) <= t['ty'] <= (y_max - edge_guard):
                _res.append(t['tx'] - (float(baseline(t['ty'])) + A_tip))
        if len(_res) >= 4:
            _res = np.asarray(_res, dtype=float)
            _mad = float(np.median(np.abs(_res - np.median(_res))))
            bent_lim = max(bent_lim, 3.0 * 1.4826 * _mad)

        n_bent = n_worn = n_missing = n_edge = 0
        for i, t in enumerate(teeth, start=1):
            t['id'] = i
            t['resid_x'] = 0.0
            if not (y_min + edge_guard) <= t['ty'] <= (y_max - edge_guard):
                t['status'] = 'EDGE'; n_edge += 1
                continue
            if not t['observed']:
                t['status'] = 'MISSING'; n_missing += 1
                continue
            # Lateral deviation from the IDEAL tip line (baseline + median amplitude)
            resid = t['tx'] - (float(baseline(t['ty'])) + A_tip)
            t['resid_x'] = resid
            if t['amp'] < worn_lim:
                # NOTE: a tooth bent INWARD also reads as low amplitude from this
                # 2-D view — WORN and "bent away from camera" are not separable here.
                t['status'] = 'WORN'; n_worn += 1
            elif abs(resid) > bent_lim:
                t['status'] = 'BENT'; n_bent += 1
            else:
                t['status'] = 'OK'

        n_damaged = n_bent + n_worn + n_missing
        if LATTICE_MAX_DAMAGED and n_damaged > LATTICE_MAX_DAMAGED:
            print(f"⚠ Blade rejected: {n_damaged} damaged teeth "
                  f"(limit {LATTICE_MAX_DAMAGED}) — bent {n_bent}, worn {n_worn}, "
                  f"missing {n_missing}")
            return None

        # ── 6. Valleys = lattice + P/2 ────────────────────────────────────────
        # X from the model, not from the (possibly deformed) neighbouring teeth:
        # resharpening should restore the INTENDED geometry, not trace the damage.
        cand_valleys = []
        for t in teeth:
            vy = t['ty'] + 0.5 * P
            if LATTICE_GRIND_IDEAL:
                vy = (y0 + round((t['ty'] - y0) / P) * P) + 0.5 * P   # ideal lattice
            vx = float(baseline(vy)) + A_val
            cand_valleys.append({'vx': vx, 'vy': vy, 'upper': t})

        # ── 7. Nearest valley to the grinder ──────────────────────────────────
        closest = None
        min_dist = float('inf')
        for c in cand_valleys:
            if (not LATTICE_GRIND_MISSING) and c['upper']['status'] == 'MISSING':
                continue
            move_x_mm = (gt_x - c['vx']) / pixels_per_mm
            move_y_mm = (gt_y - c['vy']) / pixels_per_mm
            if move_y_mm < 0.5:
                continue
            d = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            if d < min_dist:
                min_dist = d
                n_above = sum(1 for t in teeth if t['ty'] < c['vy'])
                closest = {'c': c, 'move_x_mm': move_x_mm, 'move_y_mm': move_y_mm,
                           'dist': d, 'between': f"{n_above}-{n_above + 1}"}

        # Overlay: observed tops + generated valleys
        last_overlay_pts = {
            'apexes':  [(int(t['tx']), int(t['ty'])) for t in teeth if t['observed']],
            'valleys': [(int(c['vx']), int(c['vy'])) for c in cand_valleys],
        }

        if not closest:
            return None

        confidence = round(float(min(quality, strength, snap_frac)), 3)
        return {
            'valley_id':   closest['between'],
            'x_mm':        round(float(closest['move_y_mm']), 2),   # robot X <- image Y
            'y_mm':        round(float(closest['move_x_mm']), 2),   # robot Y <- image X
            'depth_x_mm':  round(float(depth_mm), 2),
            'valley_x_px': int(closest['c']['vx']),
            'valley_y_px': int(closest['c']['vy']),
            'grinder_tip_x_px': int(gt_x),
            'grinder_tip_y_px': int(gt_y),
            'num_teeth':   int(len(teeth)),
            'distance_mm': round(float(closest['dist']), 2),
            'status': 1,
            'all_valleys': [],
            'sine_fit': False,
            # ── lattice extras ──
            'method': 'lattice',
            'pitch_mm':      round(float(P / pixels_per_mm), 4),
            'pitch_px':      round(float(P), 2),
            'pitch_quality': round(float(quality), 3),
            'phase_strength': round(float(strength), 3),
            'snap_frac':     round(float(snap_frac), 3),
            'confidence':    confidence,
            'damage': {'bent': n_bent, 'worn': n_worn, 'missing': n_missing,
                       'edge': n_edge, 'total': n_damaged},
            'teeth_status': [
                {'id': t['id'], 'status': t['status'],
                 'y_px': int(t['ty']), 'x_px': int(t['tx']),
                 'resid_mm': round(float(t['resid_x']) / pixels_per_mm, 3),
                 'observed': bool(t['observed'])}
                for t in teeth],
        }

    def analyze_frame(self, use_stored_grinder=True):
        try:
            self.preprocess_image();
            self.detect_blade_and_grinder()
            if self.blade_edge_points is None or len(self.blade_edge_points) < 20: return None
            stored = get_current_grinder_tip()
            if use_stored_grinder and stored is not None:
                self.grinder_tip = stored
            elif self.grinder_tip is not None:
                set_current_grinder_tip(self.grinder_tip)
            if not self.grinder_tip:
                return None
            if DETECTION_METHOD == 'lattice':
                # Works straight off the edge signal — no per-tooth peak hunt,
                # no dedupe pass, so it's also cheaper than the legacy path.
                return self._generate_coordinates_lattice()
            self.teeth_profiles = self.extract_tooth_profiles()
            if len(self.teeth_profiles) > 0:
                return self._generate_coordinates()
            return None
        except Exception as e:
            print(f"Analysis error: {e}");import traceback;traceback.print_exc();return None


    def _generate_coordinates(self):
        """
        NoSine / deduped method.
          1. Dedupe detected tops (keep the TALLEST per cluster), pitch-relative threshold.
          2. Fit the LINEAR ENVELOPE (+ measured pitch + A_mean) via apply_sine_correction
             on the deduped tops. The envelope smooths the lateral/X tip line; the pitch is
             MEASURED from the real tops (datasheet pitch is only a safeguard).
          3. Each valley = a reliable top anchored + 1/2 measured pitch toward the grinder
             (NOT the midpoint of two raw tops). Valley X from the envelope, depth = A_mean.
          4. Pick the valley nearest the grinder (>= 0.5 mm above it).
        The sine WAVE is never used for any value. No off-pitch rejection (yet).
        """
        global pixels_per_mm, pitch_mm, last_overlay_pts
        if len(self.teeth_profiles) < 2:
            return None
        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        if not grinder_tip:
            return None
        gt_x, gt_y = grinder_tip[0], grinder_tip[1]

        # Datasheet pitch is only a SAFEGUARD; default 2 mm if nothing set.
        nominal_pitch_mm = pitch_mm if pitch_mm > 0 else 2.0

        # 1. Dedupe tops — keep the tallest per cluster, merge anything < 0.5 pitch apart.
        deduped = _dedupe_clusters(
            self.teeth_profiles,
            y_threshold_mm=0.5 * nominal_pitch_mm,
            select='tallest',
        )
        if len(deduped) < 2:
            return None

        # 2. Linear envelope + measured pitch + A_mean (from the deduped real tops).
        corrected_apexes, _corrected_valleys, info, _above = apply_sine_correction(
            deduped, grinder_tip, pixels_per_mm, nominal_pitch_mm,
            blade_edge_points=self.blade_edge_points,
            min_teeth_for_fit=2, baseline_poly_degree=1,
        )
        if not info or not corrected_apexes or len(corrected_apexes) < 2:
            return None
        f_envelope        = np.poly1d(info['poly_coeffs'])
        measured_pitch_px = float(info['pitch_px'])              # from real tops, safeguarded to nominal
        depth_x_mm        = abs(float(info['A_mean'])) / pixels_per_mm

        apex_ys = sorted(float(a[1]) for a in corrected_apexes)

        # 3. Valley = each top + 1/2 measured pitch (anchor to the upper/reliable tooth),
        #    lateral X from the envelope.   4. Pick the nearest valley above the grinder.
        cand_valleys = []
        closest_valley = None
        min_distance = float('inf')
        for a_y in apex_ys:
            v_y = a_y + 0.5 * measured_pitch_px
            v_x = float(f_envelope(v_y))
            cand_valleys.append((v_x, v_y))
            move_x_mm = (gt_x - v_x) / pixels_per_mm
            move_y_mm = (gt_y - v_y) / pixels_per_mm
            if move_y_mm < 0.5:
                continue
            dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            if dist < min_distance:
                min_distance = dist
                n_above = sum(1 for ay in apex_ys if ay < v_y)
                closest_valley = {
                    'valley_x': v_x,
                    'valley_y': v_y,
                    'move_x_mm': move_x_mm,
                    'move_y_mm': move_y_mm,
                    'depth_x_mm': depth_x_mm,
                    'between_teeth': f"{n_above}-{n_above + 1}",
                    'distance_mm': dist,
                }

        # Store deduped tops + candidate valleys for the live overlay (debug detection).
        last_overlay_pts = {
            'apexes':  [(int(t.grinding_point[0]), int(t.grinding_point[1])) for t in deduped],
            'valleys': [(int(vx), int(vy)) for (vx, vy) in cand_valleys],
        }

        if not closest_valley:
            return None

        return {
            'valley_id': closest_valley['between_teeth'],
            'x_mm':      round(float(closest_valley['move_y_mm']), 2),
            'y_mm':      round(float(closest_valley['move_x_mm']), 2),
            'depth_x_mm': round(float(closest_valley['depth_x_mm']), 2),
            'valley_x_px': int(closest_valley['valley_x']),
            'valley_y_px': int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(gt_x),
            'grinder_tip_y_px': int(gt_y),
            'num_teeth':  int(len(deduped)),
            'distance_mm': round(float(closest_valley['distance_mm']), 2),
            'status': 1,
            'all_valleys': [],
            'sine_fit': False,
        }


# ── Inspection helpers ────────────────────────────────────────────
def _write_inspection_log(event_log, samples, blurry_samples):
    from datetime import datetime
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = 'inspection_log_' + ts + '.txt'
    hdr = '{:>4}  {:>8}  {:>10}  {:>10}  {:>8}\n'.format('#','angle','depth_mm','pitch_mm','sharpness')
    def row(s, idx):
        return '{:>4}  {:>8.2f}  {:>10.4f}  {:>10.4f}  {:>8}\n'.format(
            idx, s['angle'], s['depth_mm'], s['pitch_mm'], s['sharpness'])
    try:
        with open(filename, 'w') as f:
            f.write('MKA Blade Inspection Log -- ' + datetime.now().isoformat() + '\n')
            f.write('=' * 70 + '\n\n')
            f.write('EVENT LOG\n---------\n')
            for line in event_log:
                f.write(line + '\n')
            f.write('\nSHARP SAMPLES (blur >= threshold)\n')
            f.write('-' * 34 + '\n')
            if samples:
                f.write(hdr)
                for idx, s in enumerate(samples, 1):
                    f.write(row(s, idx))
            else:
                f.write('None.\n')
            f.write('\nBLURRY SAMPLES (blur < threshold)\n')
            f.write('-' * 34 + '\n')
            if blurry_samples:
                f.write(hdr)
                for idx, s in enumerate(blurry_samples, 1):
                    f.write(row(s, idx))
            else:
                f.write('None.\n')
            f.write('\nSUMMARY\n-------\n')
            f.write('Sharp accepted : {}\n'.format(len(samples)))
            f.write('Blurry recorded: {}\n'.format(len(blurry_samples)))
            if blurry_samples:
                t = blurry_samples[0]
                f.write('\nFIRST BLUR TRANSITION (result)\n')
                f.write('Angle      : {} deg\n'.format(t['angle']))
                f.write('Depth      : {} mm\n'.format(t['depth_mm']))
                f.write('Pitch      : {} mm\n'.format(t['pitch_mm']))
                f.write('Sharpness : {}\n'.format(t['sharpness']))
            elif samples:
                t = samples[0]
                f.write('\nFIRST SHARP SAMPLE (no transition found)\n')
                f.write('Angle      : {} deg\n'.format(t['angle']))
                f.write('Depth      : {} mm\n'.format(t['depth_mm']))
                f.write('Pitch      : {} mm\n'.format(t['pitch_mm']))
                f.write('Sharpness : {}\n'.format(t['sharpness']))
        print('Inspection log written: ' + filename)
        return filename
    except Exception as e:
        print('Could not write inspection log: ' + str(e))
        return None



# ── Inspection background logic ───────────────────────────────────────────────
def _run_inspection():
    """
    Records (angle, depth_mm, pitch_mm, sharpness) every time REG 142 changes
    by ≥ ANGLE_STEP°. No sharpness threshold — finalises by selecting the
    sample with the highest sharpness score.
    """
    global inspection_active, last_inspection_summary
    global last_frame, modbus_client, last_robot_angle

    ANGLE_STEP = 0.1   # minimum angular change (°) before taking a new sample

    samples            = []
    event_log          = []
    last_sampled_angle = None

    def _evt(tag, msg):
        ts   = time.strftime('%H:%M:%S')
        line = f"[{ts}] {tag:5s} {msg}"
        event_log.append(line)
        print(line)

    _evt('START', 'Calibration inspection recording started (angle-change gated, peak-sharpness mode)')

    while inspection_active:
        try:
            # ── 1. Read current blade angle from REG 142 ──────────────────────
            angle = last_robot_angle
            if modbus_client and modbus_client.connected:
                val = modbus_client.read_robot_angle()
                if val is not None:
                    angle = val
                    last_robot_angle = angle

            # ── 2. Skip if angle hasn't changed enough ────────────────────────
            if last_sampled_angle is not None and abs(angle - last_sampled_angle) < ANGLE_STEP:
                time.sleep(0.05)
                continue

            with camera_lock:
                if last_frame is None:
                    time.sleep(0.05)
                    continue
                frame = last_frame.copy()

            analyzer = SerratedBladeAnalyzer(frame)
            analyzer.preprocess_image()
            analyzer.detect_blade_and_grinder()
            stored = get_current_grinder_tip()
            if stored: analyzer.grinder_tip = stored
            analyzer.teeth_profiles = analyzer.extract_tooth_profiles()
            stats = analyzer._get_closest_pair_stats()  # threshold no longer matters

            if not stats:
                _evt('SKIP ', 'angle={:.2f}°  — no tooth pair detected'.format(angle))
            elif stats['sharpness_score'] is None:
                _evt('SKIP ', 'angle={:.2f}°  — sharpness unavailable'.format(angle))
            else:
                last_sampled_angle = angle
                samples.append({
                    'angle':     round(angle, 2),
                    'depth_mm':  round(stats['depth_mm'], 4),
                    'pitch_mm':  round(stats['pitch_mm'], 4),
                    'sharpness': stats['sharpness_score'],
                })
                _evt('OK   ', 'angle={:.2f}°  depth={:.4f}mm  pitch={:.4f}mm  sharp={}'.format(
                              angle, stats['depth_mm'], stats['pitch_mm'], stats['sharpness_score']))
                live = samples[-1]
                last_inspection_summary = {
                    'status':         'running',
                    'samples':        len(samples),
                    'live_angle':     live['angle'],
                    'live_depth':     live['depth_mm'],
                    'live_pitch':     live['pitch_mm'],
                    'live_sharpness': live['sharpness'],
                    'event_log':      list(event_log),
                }

        except Exception as e:
            _evt('ERR  ', 'sample error: {}'.format(e))
        time.sleep(0.1)

    # ── Finalise: pick the sample with the highest sharpness score ───────────
    if samples:
        peak = max(samples, key=lambda s: s['sharpness'])
        _evt('DONE ', 'peak_sharpness: angle={}°  depth={}mm  pitch={}mm  sharp={}  '
                      'samples={}'.format(
                      peak['angle'], peak['depth_mm'], peak['pitch_mm'],
                      peak['sharpness'], len(samples)))
        last_inspection_summary = {
            'status':         'complete',
            'samples':        len(samples),
            'peak_angle':     peak['angle'],
            'peak_depth_mm':  peak['depth_mm'],
            'peak_pitch_mm':  peak['pitch_mm'],
            'peak_sharpness': peak['sharpness'],
            'event_log':      list(event_log),
            'curve':          samples,
        }
    else:
        _evt('DONE ', 'no samples collected')
        last_inspection_summary = {
            'status':    'error',
            'message':   'No samples collected',
            'event_log': list(event_log),
        }

    _write_inspection_log(event_log, samples, [])


def _teeth_inspect_watcher():
    """
    Polls REG 141 (calibration flag) and REG 142 (robot angle) every 150 ms.
    Rising edge  (0→1): start _run_inspection thread.
    Falling edge (1→0): set inspection_active=False so thread exits cleanly.
    """
    global client_connected, modbus_client, inspection_active
    global last_inspection_summary, last_teeth_inspect_reg, last_robot_angle

    while client_connected:
        try:
            if modbus_client and modbus_client.connected:
                val   = modbus_client.read_teeth_inspect()
                angle = modbus_client.read_robot_angle()
                if angle is not None:
                    last_robot_angle = angle
                if val is not None:
                    if val and not last_teeth_inspect_reg:
                        # Rising edge → start
                        if not inspection_active and camera_active and last_frame is not None:
                            print("🔍 REG 141=1 — starting calibration inspection")
                            last_inspection_summary = {'status':'running','samples':0,
                                                       'live_angle':0,'live_depth':0,'live_pitch':0}
                            inspection_active = True
                            Thread(target=_run_inspection, daemon=True).start()
                    elif not val and last_teeth_inspect_reg:
                        # Falling edge → stop
                        if inspection_active:
                            print("🔍 REG 141=0 — stopping inspection")
                            inspection_active = False
                    last_teeth_inspect_reg = val
        except Exception:
            pass
        time.sleep(0.15)


# ── Modbus client ─────────────────────────────────────────────────────────────
class BladeDataModbusClient:
    """
    Register map:
      128  REG_BAY_ID         PC→Robot
      129  REG_GRINDER_ID     PC→Robot
      130  REG_ANGLE          PC→Robot  ×10
      131  REG_DEPTH          PC→Robot  ×100
      132  REG_LENGTH         PC→Robot  mm
      133  REG_CONFIG_VERSION PC→Robot
      134  REG_DETECTION_X    PC→Robot  ×10 signed
      135  REG_DETECTION_Y    PC→Robot  ×10 signed
      136  REG_STATUS         PC→Robot  0=none 1=ok 2=err
      137  REG_START          PC→Robot  1=start 0=idle
      138  REG_GRINDER_READY  Robot→PC  1=at position
      139  REG_GRIND_START    PC→Robot  1=grind
      140  REG_ESTOP          PC→Robot  1=HALT
      141  REG_TEETH_INSPECT  Robot→PC  1=record 0=stop
      142  REG_ROBOT_ANGLE    Robot→PC  ×10 signed degrees
      143  REG_DETECTION_DEPTH PC→Robot  ×100 signed mm  (valley depth between adjacent teeth)
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
    REG_TEETH_INSPECT  = 141
    REG_ROBOT_ANGLE    = 142
    REG_DETECTION_DEPTH = 143  # PC→Robot, X-depth of valley between teeth (×100, signed)
    REG_POST_CUT_DEPTH  = 150  # PC→Robot, measured groove depth after cut (×100 signed)

    STATUS_NO_TEETH = 0; STATUS_TEETH_OK = 1; STATUS_ERROR = 2

    def __init__(self, host='172.24.9.20', port=502, unit=1):
        self.host=host; self.port=port; self.unit=unit
        self.client=ModbusTcpClient(host, port=port); self.connected=False

    def _s16(self, v): return v if v < 32768 else v - 65536

    def connect(self):
        if self.client.connect():
            self.connected=True; print(f"✓ Connected to robot at {self.host}:{self.port}"); return True
        self.connected=False; print(f"✗ Could not connect to {self.host}:{self.port}"); return False

    def write_configuration(self, bay_id, grinder_id, angle, depth, length, config_version):
        if not self.connected: return None
        values=[int(bay_id),int(grinder_id),int(angle*10),int(depth*100),int(length),int(config_version)]
        result=self.client.write_registers(address=self.REG_BAY_ID,values=values)
        if not result.isError(): print(f"✓ Config written")
        return result

    def write_detection(self, x_mm, y_mm, status, depth_mm=0.0):
        if not self.connected: return None

        # Write depth FIRST (REG 143) so it's stable before STATUS goes high
        depth_val = int(depth_mm * 100)
        depth_u16 = depth_val if depth_val >= 0 else 65536 + depth_val
        self.client.write_register(address=self.REG_DETECTION_DEPTH, value=depth_u16)

        # Then write X/Y/STATUS contiguously (REG 134-136)
        x_val = int(x_mm * 100);
        y_val = int(y_mm * 100)
        x_u16 = x_val if x_val >= 0 else 65536 + x_val
        y_u16 = y_val if y_val >= 0 else 65536 + y_val
        result = self.client.write_registers(address=self.REG_DETECTION_X,
                                             values=[x_u16, y_u16, int(status)])
        if not result.isError():
            print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm "
                  f"depth={depth_mm:.2f}mm status={status}")
        return result

    def write_post_cut_depth(self, depth_mm):
        if not self.connected: return None
        val = int(depth_mm * 100)
        u16 = val if val >= 0 else 65536 + val
        r = self.client.write_register(address=self.REG_POST_CUT_DEPTH, value=u16)
        if not r.isError():
            print(f"✓ Post-cut depth → REG 150: {depth_mm:.2f}mm")
        return r

    def start_loop(self):
        if not self.connected: return None
        r=self.client.write_register(address=self.REG_START,value=1)
        if not r.isError(): print("✓ START=1"); return r

    def stop_loop(self):
        if not self.connected: return None
        r=self.client.write_register(address=self.REG_START,value=0)
        if not r.isError(): print("✓ START=0"); return r

    def send_grind_start(self):
        if not self.connected: return None
        r=self.client.write_register(address=self.REG_GRIND_START,value=1)
        if not r.isError(): print("✓ GRIND_START=1"); return r

    def emergency_stop(self):
        if not self.connected: return None
        self.client.write_register(address=self.REG_START,value=0)
        self.client.write_register(address=self.REG_GRIND_START,value=0)
        r=self.client.write_register(address=self.REG_ESTOP,value=1)
        if not r.isError(): print("🚨 EMERGENCY STOP SENT")
        else: print(f"✗ E-STOP failed: {r}")
        return r

    def clear_estop(self):
        if not self.connected: return None
        r=self.client.write_register(address=self.REG_ESTOP,value=0)
        if not r.isError(): print("✓ E-STOP cleared"); return r

    def read_teeth_inspect(self):
        if not self.connected: return None
        r=self.client.read_holding_registers(address=self.REG_TEETH_INSPECT,count=1)
        if r.isError(): return None
        return bool(r.registers[0])

    def read_robot_angle(self):
        """REG 142 — blade angle ×10 signed → float degrees."""
        if not self.connected: return None
        r=self.client.read_holding_registers(address=self.REG_ROBOT_ANGLE,count=1)
        if r.isError(): return None
        return self._s16(r.registers[0]) / 10.0

    def read_all_status(self):
        """Read REG 134–143 (10 registers) in one shot."""
        if not self.connected: return None
        r = self.client.read_holding_registers(address=self.REG_DETECTION_X, count=10)
        if r.isError(): return None
        reg = r.registers
        return {
            'detection_x_mm': self._s16(reg[0]) / 100.0,
            'detection_y_mm': self._s16(reg[1]) / 100.0,
            'status': reg[2],
            'start': reg[3],
            'grinder_ready': reg[4],
            'grind_start': reg[5],
            'estop': reg[6],
            'teeth_inspect': bool(reg[7]),
            'robot_angle': self._s16(reg[8]) / 10.0,
            'detection_depth_mm': self._s16(reg[9]) / 100.0,
        }

    def close(self):
        self.client.close(); self.connected=False; print("✓ Modbus connection closed")


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    try:
        with open('robot_control_dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        import os
        return f"<h1>robot_control_dashboard.html not found</h1><p>cwd:{os.getcwd()}</p>", 404

@app.route('/api/blade/<blade_id>',methods=['GET'])
def get_blade(blade_id):
    import sqlite3
    try:
        conn=sqlite3.connect("blade_database.sqlite"); conn.row_factory=sqlite3.Row
        cur=conn.cursor(); cur.execute("SELECT * FROM blades WHERE bade_id=?",(blade_id,))
        row=cur.fetchone(); conn.close()
        if row: return jsonify({"success":True,"blade":dict(row)})
        return jsonify({"success":False,"message":f"Blade '{blade_id}' not found"})
    except Exception as e: return jsonify({"success":False,"message":str(e)})

@app.route('/api/connect',methods=['POST'])
def connect():
    global modbus_client,client_connected
    data=request.json; host=data.get('host','172.24.9.20')
    port=int(data.get('port',502)); unit=int(data.get('unit',1))
    try:
        modbus_client=BladeDataModbusClient(host=host,port=port,unit=unit)
        if modbus_client.connect():
            client_connected=True
            Thread(target=_teeth_inspect_watcher,daemon=True).start()
            return jsonify({'success':True,'message':f'Connected to robot at {host}:{port}'})
        client_connected=False
        return jsonify({'success':False,'message':f'Failed to connect to {host}:{port}'}),500
    except Exception as e:
        client_connected=False; return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/disconnect',methods=['POST'])
def disconnect():
    global modbus_client,client_connected,inspection_active
    if modbus_client:
        try:
            inspection_active=False
            modbus_client.close(); modbus_client=None; client_connected=False
            return jsonify({'success':True,'message':'Disconnected'})
        except Exception as e: return jsonify({'success':False,'message':str(e)}),500
    return jsonify({'success':True,'message':'Already disconnected'})

@app.route('/api/status',methods=['GET'])
def get_status():
    return jsonify({'connected':client_connected and modbus_client is not None})

@app.route('/api/configuration', methods=['POST'])
def send_configuration():
    global current_grinder_id
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success': False, 'message': 'Not connected'}), 400
    data = request.json
    try:
        new_gid = int(data.get('grinder_id'))
        result = modbus_client.write_configuration(
            bay_id=int(data.get('bay_id')), grinder_id=new_gid,
            angle=float(data.get('angle')), depth=float(data.get('depth')),
            length=int(data.get('length')), config_version=int(data.get('config_version')))
        if not (result and not result.isError()):
            return jsonify({'success': False, 'message': 'Failed to send configuration'}), 500

        global last_commanded_depth_mm
        last_commanded_depth_mm = float(data.get('depth'))
        _reset_pitch_lock('new configuration sent')

        switched = False
        if new_gid != current_grinder_id:
            print(f"🔄 Auto-switching camera: grinder {current_grinder_id} → {new_gid}")
            was_active = camera_active
            if was_active:
                _stop_camera_internal()
            current_grinder_id = new_gid
            if was_active:
                ok, msg = _start_camera_internal(new_gid)
                if not ok:
                    return jsonify({'success': False,
                                    'message': f'Config sent but camera switch failed: {msg}'}), 500
            switched = True
        return jsonify({'success': True,
                        'message': f'Configuration sent (REG 128-133)'
                                   f'{" + camera switched" if switched else ""}',
                        'grinder_id': current_grinder_id,
                        'camera_switched': switched})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection',methods=['POST'])
def send_detection():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    data=request.json
    try:
        result=modbus_client.write_detection(x_mm=float(data.get('x_mm')),y_mm=float(data.get('y_mm')),status=int(data.get('status',1)))
        if result and not result.isError(): return jsonify({'success':True,'message':'Detection data sent'})
        return jsonify({'success':False,'message':'Failed to send detection'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/start',methods=['POST'])
def start_robot():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    try:
        result=modbus_client.start_loop()
        if result and not result.isError(): return jsonify({'success':True,'message':'Loop started (START=1)'})
        return jsonify({'success':False,'message':'Failed to start loop'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/stop',methods=['POST'])
def stop_robot():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    try:
        result=modbus_client.stop_loop()
        if result and not result.isError(): return jsonify({'success':True,'message':'Loop stopped (START=0)'})
        return jsonify({'success':False,'message':'Failed to stop loop'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/estop',methods=['POST'])
def emergency_stop():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected to robot'}),400
    try:
        result=modbus_client.emergency_stop()
        if result and not result.isError(): return jsonify({'success':True,'message':'EMERGENCY STOP — REG 140=1'})
        return jsonify({'success':False,'message':'Failed to send E-STOP'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/estop/clear',methods=['POST'])
def clear_estop():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    try:
        result=modbus_client.clear_estop()
        if result and not result.isError(): return jsonify({'success':True,'message':'E-STOP cleared (REG 140=0)'})
        return jsonify({'success':False,'message':'Failed to clear E-STOP'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/grind_start',methods=['POST'])
def grind_start():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    data=request.json or {}
    try:
        if 'x_mm' in data and 'y_mm' in data:
            modbus_client.write_detection(
                x_mm=float(data['x_mm']),
                y_mm=float(data['y_mm']),
                status=int(data.get('status', 1)),
                depth_mm=float(data.get('depth_mm', 0.0)),
            )
            global last_grind_depth_mm
            last_grind_depth_mm = float(data.get('depth_mm', 0.0))
        result=modbus_client.send_grind_start()
        if result and not result.isError(): return jsonify({'success':True,'message':'GRIND_START=1 sent'})
        return jsonify({'success':False,'message':'Failed to send GRIND_START'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/robot_registers',methods=['GET'])
def robot_registers():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    try:
        status=modbus_client.read_all_status()
        if status: return jsonify({'success':True,'registers':status})
        return jsonify({'success':False,'message':'Failed to read registers'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/camera/ready', methods=['GET'])
def camera_ready():
    return jsonify({'ready': camera_active and last_frame is not None})

# ── Inspection endpoints ──────────────────────────────────────────────────────
@app.route('/api/inspection/start',methods=['POST'])
def start_inspection():
    global inspection_active,last_inspection_summary
    if not camera_active or last_frame is None:
        return jsonify({'success':False,'message':'Camera not active'}),400
    if inspection_active:
        return jsonify({'success':False,'message':'Inspection already running'}),400
    last_inspection_summary={'status':'running','samples':0,'live_angle':0,'live_depth':0,'live_pitch':0}
    inspection_active=True
    Thread(target=_run_inspection,daemon=True).start()
    return jsonify({'success':True,'message':'Inspection started'})

@app.route('/api/inspection/stop',methods=['POST'])
def stop_inspection():
    global inspection_active
    if not inspection_active:
        return jsonify({'success':False,'message':'Inspection not running'}),400
    inspection_active=False
    return jsonify({'success':True,'message':'Inspection stopping…'})

@app.route('/api/inspection/status',methods=['GET'])
def inspection_status_route():
    return jsonify({'active':inspection_active,'result':last_inspection_summary,'current_angle':last_robot_angle})


# ── PySpin system management (KEEP ALIVE for the whole process) ───────────────
# System.GetInstance() + GetCameras() triggers a full Spinnaker device
# discovery, which on Windows can take 30-40+ seconds (it probes every
# network interface). We therefore do it ONCE and keep both the System and
# the CameraList alive. Camera switches only DeInit/Init the camera object.

def _get_pyspin_system(refresh=False):
    """
    Initialise Spinnaker once and keep it alive for the whole process.
    refresh=True re-enumerates the camera list (only safe while no camera
    is acquiring) — used if cameras were plugged in after startup.
    """
    global pyspin_system, pyspin_cam_list
    if pyspin_system is None:
        print("🎥 Initialising Spinnaker system (one-time discovery — may take a while)…")
        t0 = time.perf_counter()
        pyspin_system = PySpin.System.GetInstance()
        pyspin_cam_list = pyspin_system.GetCameras()
        print(f"✓ Spinnaker ready in {time.perf_counter()-t0:.1f}s "
              f"({pyspin_cam_list.GetSize()} camera(s) found)")
    elif refresh and not camera_active:
        try:
            pyspin_cam_list.Clear()
        except Exception:
            pass
        pyspin_cam_list = pyspin_system.GetCameras()
        print(f"✓ Camera list refreshed ({pyspin_cam_list.GetSize()} camera(s))")
    return pyspin_system, pyspin_cam_list


@atexit.register
def _release_pyspin():
    """Release Spinnaker cleanly when the process exits."""
    global pyspin_cam, pyspin_cam_list, pyspin_system
    try:
        if pyspin_cam:
            try:
                pyspin_cam.EndAcquisition()
            except Exception:
                pass
            try:
                pyspin_cam.DeInit()
            except Exception:
                pass
            pyspin_cam = None
        if pyspin_cam_list is not None:
            try:
                pyspin_cam_list.Clear()
            except Exception:
                pass
            pyspin_cam_list = None
        if pyspin_system is not None:
            try:
                pyspin_system.ReleaseInstance()
            except Exception:
                pass
            pyspin_system = None
        print("✓ Spinnaker released")
    except Exception:
        pass


# ── Camera ────────────────────────────────────────────────────────────────────
def _stop_camera_internal():
    """Stop capture and DeInit the camera. The PySpin System and CameraList
    are intentionally KEPT ALIVE so the next start/switch is fast."""
    global pyspin_cam, camera_active, camera_thread, last_frame
    camera_active = False
    th = camera_thread
    camera_thread = None
    if th and th.is_alive():
        th.join(timeout=2.0)
    with camera_lock:
        if pyspin_cam:
            try:
                pyspin_cam.EndAcquisition()
            except Exception:
                pass
            try:
                pyspin_cam.DeInit()
            except Exception:
                pass
            pyspin_cam = None
        last_frame = None
        # NOTE: system + cam_list intentionally kept alive (see _get_pyspin_system)


def _start_camera_internal(grinder_id):
    """Initialise and start PySpin camera for the given grinder id. Returns (ok, msg).
    Uses the persistent system/camera list — no Spinnaker re-discovery."""
    global pyspin_cam, camera_active, camera_thread
    if camera_active:
        return False, 'Camera already running (stop first)'
    serial = CAMERA_SERIALS.get(grinder_id, '')
    print(f"🎥 Starting FLIR camera for grinder {grinder_id} "
          f"(serial='{serial or 'auto-index ' + str(grinder_id - 1)}')")

    system, cam_list = _get_pyspin_system()
    n = cam_list.GetSize()
    if n == 0:
        # Maybe cameras were plugged in after startup — refresh the list once
        system, cam_list = _get_pyspin_system(refresh=True)
        n = cam_list.GetSize()
        if n == 0:
            return False, 'No FLIR cameras detected'

    selected = None
    selected_idx = None
    if serial:
        for i in range(n):
            cam = cam_list[i]
            cam_sn = ''
            try:
                tldev = cam.GetTLDeviceNodeMap()
                sn_node = PySpin.CStringPtr(tldev.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(sn_node):
                    cam_sn = sn_node.GetValue()
                del sn_node, tldev
            except Exception:
                cam_sn = ''
            if cam_sn == serial:
                selected_idx = i
                del cam
                break
            del cam
        if selected_idx is None:
            return False, f"Camera with serial '{serial}' not found for grinder {grinder_id}"
        selected = cam_list[selected_idx]
    else:
        idx = grinder_id - 1
        if not (0 <= idx < n):
            return False, f'Camera index {idx} out of range (have {n})'
        selected = cam_list[idx]

    try:
        pyspin_cam = selected
        pyspin_cam.Init()
        _configure_pyspin_camera()
        pyspin_cam.BeginAcquisition()
    except PySpin.SpinnakerException as ex:
        # Clean up the half-initialised camera so a retry is possible
        try:
            if pyspin_cam:
                pyspin_cam.DeInit()
        except Exception:
            pass
        pyspin_cam = None
        return False, f'Camera init failed: {ex}'

    camera_active = True
    camera_thread = Thread(target=_camera_capture_thread, daemon=True)
    camera_thread.start()
    print(f"✓ FLIR camera started for grinder {grinder_id}")
    return True, f'Camera started for grinder {grinder_id}'


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    global current_grinder_id
    data = request.json or {}
    if 'grinder_id' in data:
        current_grinder_id = int(data['grinder_id'])
    ok, msg = _start_camera_internal(current_grinder_id)
    if ok:
        return jsonify({'success': True, 'message': msg, 'grinder_id': current_grinder_id})
    return jsonify({'success': False, 'message': msg}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    try:
        _stop_camera_internal()
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/switch', methods=['POST'])
def switch_camera():
    """Switch active grinder/camera. Stops + restarts camera if it was running."""
    global current_grinder_id
    data = request.json or {}
    new_gid = int(data.get('grinder_id', current_grinder_id))
    if new_gid not in CAMERA_SERIALS:
        return jsonify({'success': False, 'message': f'Unknown grinder_id {new_gid}'}), 400
    was_active = camera_active
    if was_active:
        _stop_camera_internal()
    current_grinder_id = new_gid
    if was_active:
        ok, msg = _start_camera_internal(new_gid)
        if not ok:
            return jsonify({'success': False, 'message': f'Switch failed: {msg}',
                            'grinder_id': new_gid}), 500
        return jsonify({'success': True, 'grinder_id': new_gid,
                        'message': f'Switched to grinder {new_gid} (camera restarted)'})
    return jsonify({'success': True, 'grinder_id': new_gid,
                    'message': f'Switched to grinder {new_gid} (camera was off)'})

@app.route('/api/camera/list', methods=['GET'])
def list_cameras():
    """Enumerate detected FLIR cameras. Camera must be stopped to (re)enumerate.
    Uses the persistent Spinnaker system — no full re-discovery."""
    if camera_active:
        return jsonify({'success': False, 'message': 'Stop camera first to list devices',
                        'mapping': CAMERA_SERIALS,
                        'current_grinder_id': current_grinder_id}), 400
    try:
        # Refresh the persistent list (safe: camera is not active)
        system, cl = _get_pyspin_system(refresh=True)
        cams = []
        n = cl.GetSize()
        for i in range(n):
            cam = cl[i]
            sn_value = 'unknown'
            md_value = 'unknown'
            try:
                tldev = cam.GetTLDeviceNodeMap()
                sn = PySpin.CStringPtr(tldev.GetNode('DeviceSerialNumber'))
                md = PySpin.CStringPtr(tldev.GetNode('DeviceModelName'))
                if PySpin.IsReadable(sn): sn_value = sn.GetValue()
                if PySpin.IsReadable(md): md_value = md.GetValue()
                # Drop refs to nodes/nodemap BEFORE the camera goes out of scope
                del sn, md, tldev
            except Exception as e:
                sn_value = 'error'
                md_value = str(e)
            cams.append({'index': i, 'serial': sn_value, 'model': md_value})
            # Drop the camera ref every iteration — critical for -1004 fix
            del cam
        return jsonify({'success': True, 'cameras': cams,
                        'mapping': CAMERA_SERIALS,
                        'current_grinder_id': current_grinder_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/camera/mapping', methods=['POST'])
def set_camera_mapping():
    """Update {grinder_id: serial} map and persist to disk."""
    global CAMERA_SERIALS
    data = request.json or {}
    try:
        new_map = {int(k): str(v) for k, v in (data.get('mapping') or {}).items()}
        CAMERA_SERIALS.update(new_map)
        save_camera_serials()
        return jsonify({'success': True, 'mapping': CAMERA_SERIALS})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/grinder/current', methods=['GET'])
def get_current_grinder():
    return jsonify({'success': True,
                    'current_grinder_id': current_grinder_id,
                    'tip': (list(get_current_grinder_tip()) if get_current_grinder_tip() else None),
                    'all_tips': {k: list(v) for k, v in stored_grinder_tips.items()},
                    'mapping': CAMERA_SERIALS})

def _configure_pyspin_camera():
    global pyspin_cam,camera_config
    try:
        nodemap=pyspin_cam.GetNodeMap()
        node_acq=PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if PySpin.IsWritable(node_acq): node_acq.SetIntValue(node_acq.GetEntryByName('Continuous').GetValue())
        node_fr_en=PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsWritable(node_fr_en): node_fr_en.SetValue(True)
        node_fr=PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if PySpin.IsWritable(node_fr): node_fr.SetValue(min(node_fr.GetMax(),camera_config['frame_rate']))
        # node_exp_auto=PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        # if PySpin.IsWritable(node_exp_auto): node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
        # node_exp=PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        # if PySpin.IsWritable(node_exp): node_exp.SetValue(min(node_exp.GetMax(),camera_config['exposure_time']))
        node_gain_auto=PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsWritable(node_gain_auto): node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
        # node_gain=PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        # if PySpin.IsWritable(node_gain): node_gain.SetValue(min(node_gain.GetMax(),camera_config['gain']))
    except PySpin.SpinnakerException as ex: print(f"⚠ Camera config warning: {ex}")


# ── Camera settings (live tuning: exposure / gain / white balance / fps …) ────
# These read and write Spinnaker feature nodes on the running camera. Reads are
# open to viewers; writes go through the control lock (POST is gated) so only the
# controller can change the shared camera. Node names follow the GenICam SFNC
# standard used by FLIR/Teledyne cameras.

def _cam_read_enum(nm, name):
    try:
        node = PySpin.CEnumerationPtr(nm.GetNode(name))
        if not PySpin.IsReadable(node):
            return None
        entries = []
        for i in range(node.GetNumEntries()):
            e = PySpin.CEnumEntryPtr(node.GetEntryByIndex(i))
            if PySpin.IsAvailable(e) and PySpin.IsReadable(e):
                entries.append(e.GetSymbolic())
        cur = PySpin.CEnumEntryPtr(node.GetCurrentEntry())
        return {'value': cur.GetSymbolic() if PySpin.IsReadable(cur) else None,
                'entries': entries,
                'writable': bool(PySpin.IsWritable(node))}
    except Exception:
        return None

def _cam_read_float(nm, name):
    try:
        node = PySpin.CFloatPtr(nm.GetNode(name))
        if not PySpin.IsReadable(node):
            return None
        return {'value': round(float(node.GetValue()), 3),
                'min': round(float(node.GetMin()), 3),
                'max': round(float(node.GetMax()), 3),
                'writable': bool(PySpin.IsWritable(node))}
    except Exception:
        return None

def _cam_read_bool(nm, name):
    try:
        node = PySpin.CBooleanPtr(nm.GetNode(name))
        if not PySpin.IsReadable(node):
            return None
        return {'value': bool(node.GetValue()),
                'writable': bool(PySpin.IsWritable(node))}
    except Exception:
        return None

def _cam_set_enum(nm, name, symbolic):
    node = PySpin.CEnumerationPtr(nm.GetNode(name))
    if not PySpin.IsWritable(node):
        raise RuntimeError(f'{name} not writable')
    entry = PySpin.CEnumEntryPtr(node.GetEntryByName(str(symbolic)))
    if not PySpin.IsReadable(entry):
        raise RuntimeError(f'{name}={symbolic} not available')
    node.SetIntValue(entry.GetValue())

def _cam_set_float(nm, name, value):
    node = PySpin.CFloatPtr(nm.GetNode(name))
    if not PySpin.IsWritable(node):
        raise RuntimeError(f'{name} not writable (is its Auto mode still on?)')
    v = max(node.GetMin(), min(node.GetMax(), float(value)))
    node.SetValue(v)

def _cam_set_bool(nm, name, value):
    node = PySpin.CBooleanPtr(nm.GetNode(name))
    if not PySpin.IsWritable(node):
        raise RuntimeError(f'{name} not writable')
    node.SetValue(bool(value))

def _get_camera_settings():
    global pyspin_cam
    if not (camera_active and pyspin_cam):
        return {'success': False, 'message': 'Camera not active'}
    with camera_lock:
        try:
            nm = pyspin_cam.GetNodeMap()
            return {
                'success': True,
                'exposure_auto':      _cam_read_enum(nm, 'ExposureAuto'),
                'exposure_time':      _cam_read_float(nm, 'ExposureTime'),   # microseconds
                'gain_auto':          _cam_read_enum(nm, 'GainAuto'),
                'gain':               _cam_read_float(nm, 'Gain'),           # dB
                'white_balance_auto': _cam_read_enum(nm, 'BalanceWhiteAuto'),# color cams only
                'frame_rate_enable':  _cam_read_bool(nm, 'AcquisitionFrameRateEnable'),
                'frame_rate':         _cam_read_float(nm, 'AcquisitionFrameRate'),
                'gamma_enable':       _cam_read_bool(nm, 'GammaEnable'),
                'gamma':              _cam_read_float(nm, 'Gamma'),
                'black_level':        _cam_read_float(nm, 'BlackLevel'),
                'grinder_id':         current_grinder_id,
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

@app.route('/api/camera/settings', methods=['GET'])
def camera_settings_get():
    return jsonify(_get_camera_settings())

@app.route('/api/camera/settings', methods=['POST'])
def camera_settings_set():
    from flask import request
    global pyspin_cam, camera_config
    if not (camera_active and pyspin_cam):
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    data = request.json or {}
    applied, errors = [], []

    def _try(label, fn):
        try:
            fn(); applied.append(label)
        except Exception as e:
            errors.append(f'{label}: {e}')

    def _is_auto(v):
        return str(v) in ('Continuous', 'Once')

    with camera_lock:
        try:
            nm = pyspin_cam.GetNodeMap()
        except Exception as e:
            return jsonify({'success': False, 'message': f'nodemap error: {e}'}), 500

        # Order matters: set Auto modes BEFORE their manual values, because
        # ExposureTime/Gain are only writable once their Auto is Off.
        if 'exposure_auto' in data:
            _try('exposure_auto', lambda: _cam_set_enum(nm, 'ExposureAuto', data['exposure_auto']))
        if data.get('exposure_time') is not None and not _is_auto(data.get('exposure_auto')):
            _try('exposure_time', lambda: _cam_set_float(nm, 'ExposureTime', data['exposure_time']))

        if 'gain_auto' in data:
            _try('gain_auto', lambda: _cam_set_enum(nm, 'GainAuto', data['gain_auto']))
        if data.get('gain') is not None and not _is_auto(data.get('gain_auto')):
            _try('gain', lambda: _cam_set_float(nm, 'Gain', data['gain']))

        if 'white_balance_auto' in data:
            _try('white_balance_auto', lambda: _cam_set_enum(nm, 'BalanceWhiteAuto', data['white_balance_auto']))

        # Frame rate: must enable manual frame-rate control before setting it.
        if 'frame_rate_enable' in data:
            _try('frame_rate_enable', lambda: _cam_set_bool(nm, 'AcquisitionFrameRateEnable', data['frame_rate_enable']))
        if data.get('frame_rate') is not None and data.get('frame_rate_enable', True):
            _try('frame_rate', lambda: _cam_set_float(nm, 'AcquisitionFrameRate', data['frame_rate']))

        if 'gamma_enable' in data:
            _try('gamma_enable', lambda: _cam_set_bool(nm, 'GammaEnable', data['gamma_enable']))
        if data.get('gamma') is not None:
            _try('gamma', lambda: _cam_set_float(nm, 'Gamma', data['gamma']))

        if data.get('black_level') is not None:
            _try('black_level', lambda: _cam_set_float(nm, 'BlackLevel', data['black_level']))

    # Persist frame rate so a camera restart keeps it.
    if data.get('frame_rate') is not None:
        try: camera_config['frame_rate'] = float(data['frame_rate'])
        except Exception: pass

    fresh = _get_camera_settings()
    print(f"🎛  Camera settings applied={applied}" + (f" errors={errors}" if errors else ""))
    return jsonify({'success': len(errors) == 0, 'applied': applied,
                    'errors': errors, 'settings': fresh})


def _camera_capture_thread():
    global pyspin_cam,camera_active,last_frame
    processor=PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
    while camera_active:
        try:
            image_result=pyspin_cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                pf=image_result.GetPixelFormat()
                if pf==PySpin.PixelFormat_BGR8: frame=image_result.GetNDArray()
                elif pf==PySpin.PixelFormat_Mono8: frame=cv2.cvtColor(image_result.GetNDArray(),cv2.COLOR_GRAY2BGR)
                else: frame=processor.Convert(image_result,PySpin.PixelFormat_BGR8).GetNDArray()
                frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
                with camera_lock: last_frame=frame.copy()
                image_result.Release()
        except PySpin.SpinnakerException: time.sleep(0.01)
        except Exception as e:
            if camera_active: print(f"⚠ Frame error: {e}")
            time.sleep(0.01)

def draw_detection_overlay(frame,detection_result):
    overlay=frame.copy()
    if not detection_result: return overlay
    try:
        gt=(detection_result.get('grinder_tip_x_px'),detection_result.get('grinder_tip_y_px'))
        if gt and gt[0]>0:
            gti=(int(gt[0]),int(gt[1]))
            cv2.circle(overlay,gti,12,(0,255,255),3)
            cv2.putText(overlay,"GRINDER",(gti[0]+20,gti[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        vx,vy=detection_result.get('valley_x_px'),detection_result.get('valley_y_px')
        if vx and vy:
            vp=(int(vx),int(vy)); cv2.circle(overlay,vp,10,(255,0,255),-1); cv2.circle(overlay,vp,12,(255,255,255),2)
            cv2.putText(overlay,f"({detection_result.get('x_mm',0):+.1f},{detection_result.get('y_mm',0):+.1f})mm",(vp[0]-35,vp[1]-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)
            if gt and gt[0]>0: cv2.arrowedLine(overlay,vp,(int(gt[0]),int(gt[1])),(0,255,255),2,tipLength=0.02)
        # NoSine overlay: deduped tops (green stars) + candidate valleys (blue dots)
        if last_overlay_pts:
            for (ax, ay) in last_overlay_pts.get('apexes', []):
                cv2.drawMarker(overlay, (int(ax), int(ay)), (0, 255, 0),
                               markerType=cv2.MARKER_STAR, markerSize=12, thickness=2)
            for (cvx, cvy) in last_overlay_pts.get('valleys', []):
                cv2.circle(overlay, (int(cvx), int(cvy)), 5, (255, 128, 0), -1)
        # Sine envelope is drawn from the fresh per-frame fit in get_camera_frame (gated by the
        # dashboard 'sine' overlay mode via sine_overlay_enabled), not from the stale loop fit here.
    except Exception as e: print(f"Overlay error: {e}")
    return overlay

@app.route('/api/camera/frame')
def get_camera_frame():
    global camera_active,last_frame,last_detection_result
    if not camera_active or last_frame is None:
        blank=np.zeros((480,640,3),dtype=np.uint8); _,buf=cv2.imencode('.jpg',blank)
        return Response(buf.tobytes(),mimetype='image/jpeg')
    try:
        with camera_lock: frame=last_frame.copy() if last_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        if last_detection_result: frame=draw_detection_overlay(frame,last_detection_result)
        if sine_overlay_enabled:
            if live_sine_overlay and (time.time()-live_sine_overlay.get('t',0))<1.0:
                try:
                    draw_sine_overlay(frame,live_sine_overlay['apexes'],
                                      live_sine_overlay['valleys'],live_sine_overlay['info'])
                except Exception as _e:
                    print(f"[sine_overlay] live draw error: {_e}")
            else:
                hint=('SINE: set pitch > 0 to enable fit' if pitch_mm<=0
                      else 'SINE: waiting for fit (need grinder tip + teeth)')
                cv2.putText(frame,hint,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,128),2)
        ret,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        return Response(buf.tobytes() if ret else np.zeros((480,640,3),dtype=np.uint8),mimetype='image/jpeg')
    except:
        blank=np.zeros((480,640,3),dtype=np.uint8); _,buf=cv2.imencode('.jpg',blank)
        return Response(buf.tobytes(),mimetype='image/jpeg')

@app.route('/api/camera/capture',methods=['POST'])
def capture_frame():
    if not camera_active or last_frame is None:
        return jsonify({'success':False,'message':'Camera not active'}),400
    try:
        with camera_lock: frame=last_frame.copy()
        filename=f'capture_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename,frame)   # explicit user action — keep the actual save
        return jsonify({'success':True,'message':'Captured','filename':filename})
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/analyze',methods=['POST'])
def analyze_current_frame():
    global camera_active,last_frame,last_detection_result
    if not camera_active or last_frame is None:
        return jsonify({'success':False,'message':'Camera not active'}),400
    try:
        with camera_lock: frame_to_process=last_frame.copy()
        analyzer=SerratedBladeAnalyzer(frame_to_process); result=analyzer.analyze_frame()
        if result:
            last_detection_result=result
            return jsonify({'success':True,'detection':result,'message':f"Valley {result.get('valley_id','N/A')}"})
        return jsonify({'success':False,'message':'No teeth detected'}),404
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/measure_post_cut', methods=['POST'])
def measure_post_cut():
    """Triggered on GRIND_START 1→0. Measures the just-cut valley's depth,
    writes it to REG 150, and returns measured vs expected for validation."""
    global last_post_cut_result
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    data = request.json or {}
    settle_ms = int(data.get('settle_ms', 150))
    time.sleep(settle_ms / 1000.0)   # let grinder retract / frame settle
    try:
        with camera_lock:
            frame = last_frame.copy()
        an = SerratedBladeAnalyzer(frame)
        an.preprocess_image(); an.detect_blade_and_grinder()
        stored = get_current_grinder_tip()
        if stored: an.grinder_tip = stored
        an.teeth_profiles = an.extract_tooth_profiles()
        meas = an.measure_grinder_valley_depth()
        if not meas:
            return jsonify({'success': False,
                            'message': 'Could not measure post-cut valley'}), 404

        commanded = last_commanded_depth_mm
        pre_cut   = last_grind_depth_mm
        measured  = meas['depth_mm']
        # Expected FINAL valley depth = commanded depth (REG 131), measured from the
        # tooth-tip line. The commanded value IS the target total depth, not an
        # increment on the existing groove — so expected = commanded, NOT pre_cut + commanded.
        expected  = commanded
        delta     = (measured - expected) if expected is not None else None

        if modbus_client and modbus_client.connected:
            modbus_client.write_post_cut_depth(measured)

        last_post_cut_result = {
            'measured_depth_mm':  measured,
            'commanded_depth_mm': commanded,
            'pre_cut_depth_mm':   pre_cut,
            'expected_depth_mm':  round(expected, 3) if expected is not None else None,
            'delta_mm':           round(delta, 3) if delta is not None else None,
            'valley_id':          meas['valley_id'],
            'valley_x_px':        meas['valley_x_px'],
            'valley_y_px':        meas['valley_y_px'],
            'ts':                 time.strftime('%H:%M:%S'),
        }
        return jsonify({'success': True, 'result': last_post_cut_result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/detection/send_auto',methods=['POST'])
def send_auto_detection():
    global camera_active,last_frame,modbus_client
    if not camera_active or last_frame is None:
        return jsonify({'success':False,'message':'Camera not active'}),400
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    try:
        with camera_lock: frame_to_process=last_frame.copy()
        analyzer=SerratedBladeAnalyzer(frame_to_process); result=analyzer.analyze_frame()
        if not result: return jsonify({'success':False,'message':'No teeth detected'}),404
        modbus_result = modbus_client.write_detection(
            x_mm=result['x_mm'],
            y_mm=result['y_mm'],
            status=result['status'],
            depth_mm=result.get('depth_x_mm', 0.0),
        )
        if modbus_result and not modbus_result.isError():
            return jsonify({'success':True,'detection':result,'message':f"Valley {result.get('valley_id','N/A')} sent"})
        return jsonify({'success':False,'message':'Failed to send to robot'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/status',methods=['GET'])
def get_detection_status():
    return jsonify({'enabled':detection_enabled,'last_result':last_detection_result,'pixels_per_mm':pixels_per_mm,'pitch_mm':pitch_mm,'sine_overlay':sine_overlay_enabled})


@app.route('/api/detection/sine_overlay', methods=['GET', 'POST'])
def set_sine_overlay():
    global sine_overlay_enabled
    if request.method == 'POST':
        data = request.json or {}
        sine_overlay_enabled = bool(data.get('enabled', False))
        print(f"[sine_overlay] live envelope {'SHOWN' if sine_overlay_enabled else 'HIDDEN'}")
    return jsonify({'success': True, 'enabled': sine_overlay_enabled})

@app.route('/api/detection/pitch', methods=['POST'])
def set_pitch():
    global pitch_mm
    data = request.json or {}
    val  = data.get('pitch_mm', 0)
    try:
        pitch_mm = max(0.0, float(val))
        _reset_pitch_lock('pitch changed')
        msg = (f'Pitch = {pitch_mm:.3f} mm — '
               f"sine-fit {'ENABLED' if pitch_mm > 0 else 'DISABLED'}")
        print(f'[pitch] {msg}')
        return jsonify({'success': True, 'pitch_mm': pitch_mm, 'message': msg})
    except (TypeError, ValueError) as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/detection/method', methods=['GET', 'POST'])
def detection_method():
    """Switch between the lattice and legacy detectors live (for A/B on the machine)."""
    global DETECTION_METHOD
    if request.method == 'POST':
        data = request.json or {}
        m = str(data.get('method', '')).lower()
        if m not in ('lattice', 'legacy'):
            return jsonify({'success': False,
                            'message': "method must be 'lattice' or 'legacy'"}), 400
        DETECTION_METHOD = m
        _reset_pitch_lock('method changed')
        print(f"🔬 Detection method → {m}")
    return jsonify({'success': True, 'method': DETECTION_METHOD,
                    'pitch_lock': {'pitch_px': _pitch_lock['pitch_px'],
                                   'pitch_mm': (_pitch_lock['pitch_px'] / pixels_per_mm
                                                if _pitch_lock['pitch_px'] else None),
                                   'quality': _pitch_lock['quality']}})


@app.route('/api/detection/reset_pitch_lock', methods=['POST'])
def reset_pitch_lock_route():
    _reset_pitch_lock('manual reset')
    return jsonify({'success': True, 'message': 'Pitch lock cleared — will re-measure'})

@app.route('/api/detection/calibrate',methods=['POST'])
def calibrate_detection():
    global pixels_per_mm
    data=request.json; new_ppm=data.get('pixels_per_mm')
    if new_ppm and new_ppm>0:
        pixels_per_mm=float(new_ppm); return jsonify({'success':True,'pixels_per_mm':pixels_per_mm})
    return jsonify({'success':False,'message':'Invalid value'}),400

@app.route('/api/detection/update_grinder', methods=['POST'])
def update_grinder_position():
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    try:
        with camera_lock:
            frame_to_process = last_frame.copy()
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        analyzer.analyze_frame(use_stored_grinder=False)
        if analyzer.grinder_tip:
            set_current_grinder_tip(analyzer.grinder_tip)
            tip = get_current_grinder_tip()
            return jsonify({'success': True,
                            'grinder_id': current_grinder_id,
                            'grinder_tip': {'x': int(tip[0]), 'y': int(tip[1])},
                            'message': f'Grinder {current_grinder_id} tip updated: {tip}'})
        return jsonify({'success': False, 'message': 'Could not detect grinder'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def _capture_grinder_average(duration_s=2.0, settle_ms=20):
    """
    Record grinder tip detections for `duration_s` and return a robust average.
    The grinder vibrates laterally during operation, so a single-frame
    'SET GRINDER' can land off-centre. Averaging ~50 frames and rejecting
    gross V-fit failures (MAD on distance-from-median) gives a stable tip.
    """
    global last_frame
    if not camera_active or last_frame is None:
        return {'success': False, 'message': 'Camera not active'}

    tips = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        with camera_lock:
            frame = last_frame.copy() if last_frame is not None else None
        if frame is not None:
            try:
                analyzer = SerratedBladeAnalyzer(frame)
                analyzer.preprocess_image()
                analyzer.detect_blade_and_grinder()
                if analyzer.grinder_tip is not None:
                    tips.append((float(analyzer.grinder_tip[0]),
                                 float(analyzer.grinder_tip[1])))
            except Exception:
                pass
        time.sleep(settle_ms / 1000.0)

    if len(tips) < 3:
        return {'success': False,
                'message': f'Too few valid detections ({len(tips)}) — check lighting/grinder'}

    pts = np.array(tips, dtype=float)
    med = np.median(pts, axis=0)
    d   = np.linalg.norm(pts - med, axis=1)            # distance of each point from median
    mad = np.median(np.abs(d - np.median(d))) or 1.0   # robust spread of those distances
    keep = d <= (np.median(d) + 3.0 * 1.4826 * mad)    # 3σ-equivalent gate
    kept = pts[keep] if keep.sum() >= 3 else pts

    avg = kept.mean(axis=0)
    tip = (int(round(avg[0])), int(round(avg[1])))
    set_current_grinder_tip(tip)

    return {
        'success': True,
        'tip': {'x': tip[0], 'y': tip[1]},
        'samples':  int(len(tips)),
        'kept':     int(keep.sum()),
        'rejected': int(len(tips) - int(keep.sum())),
        'x_std':    round(float(kept[:, 0].std()), 2),
        'y_std':    round(float(kept[:, 1].std()), 2),
        'grinder_id': current_grinder_id,
    }

@app.route('/api/detection/set_grinder_manual', methods=['POST'])
def set_grinder_manual():
    data = request.json or {}
    try:
        x = int(round(float(data['x'])))
        y = int(round(float(data['y'])))
    except (KeyError, TypeError, ValueError):
        return jsonify({'success': False, 'message': 'Provide numeric x and y'}), 400
    set_current_grinder_tip((x, y))
    return jsonify({'success': True, 'grinder_id': current_grinder_id,
                    'grinder_tip': {'x': x, 'y': y},
                    'message': f'Grinder {current_grinder_id} tip set manually: ({x}, {y})'})


@app.route('/api/detection/set_grinder_average', methods=['POST'])
def set_grinder_average():
    if not camera_active or last_frame is None:
        return jsonify({'success': False, 'message': 'Camera not active'}), 400
    data = request.json or {}
    try:
        duration = max(0.5, min(float(data.get('duration_s', 2.0)), 10.0))
    except (TypeError, ValueError):
        duration = 2.0
    result = _capture_grinder_average(duration_s=duration)
    return jsonify(result), (200 if result.get('success') else 404)

@app.route('/api/detection/teeth_profiles', methods=['GET'])
def get_teeth_profiles():
    try:
        frame_to_process = None
        if camera_active and last_frame is not None:
            with camera_lock: frame_to_process = last_frame.copy()
        stored = get_current_grinder_tip()
        if frame_to_process is None:
            grinder = {'x': int(stored[0]), 'y': int(stored[1])} if stored else None
            return jsonify({'success': True, 'teeth': [], 'grinder_tip': grinder, 'num_teeth': 0,
                            'grinder_id': current_grinder_id})
        analyzer = SerratedBladeAnalyzer(frame_to_process)
        analyzer.preprocess_image(); analyzer.detect_blade_and_grinder()
        analyzer.draw_grinder_v_overlay(frame_to_process)

        profiles = analyzer.extract_tooth_profiles()
        grinder_tip = None
        if pitch_mm > 0 and stored:
            global live_sine_overlay
            _apx, _vly, _info, _ = apply_sine_correction(
                profiles, stored, pixels_per_mm, pitch_mm,
                blade_edge_points=analyzer.blade_edge_points,
                min_teeth_for_fit=2,
                baseline_poly_degree=1)
            if _info:
                # Cache the fresh fit so the live feed (get_camera_frame) can draw it
                # when the dashboard overlay selector is in 'sine' mode.
                live_sine_overlay = {'apexes': _apx, 'valleys': _vly,
                                     'info': _info, 't': time.time()}
        if stored: analyzer.grinder_tip = stored
        if analyzer.grinder_tip:
            grinder_tip = {'x': int(analyzer.grinder_tip[0]), 'y': int(analyzer.grinder_tip[1])}
        elif stored:
            grinder_tip = {'x': int(stored[0]), 'y': int(stored[1])}
        teeth = [{'tooth_id': t.tooth_id,
                  'tip': {'x': int(t.grinding_point[0]), 'y': int(t.grinding_point[1])},
                  'top_valley': {'x': int(t.top_valley[0]), 'y': int(t.top_valley[1])},
                  'bottom_valley': {'x': int(t.bottom_valley[0]), 'y': int(t.bottom_valley[1])},
                  'angle': round(float(t.angle), 2), 'height': round(float(t.height), 2)}
                 for t in profiles]
        return jsonify({'success': True, 'teeth': teeth, 'grinder_tip': grinder_tip,
                        'num_teeth': len(teeth), 'grinder_id': current_grinder_id})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/detection/grinder_status', methods=['GET'])
def get_grinder_status():
    tip = get_current_grinder_tip()
    return jsonify({
        'success': True,
        'grinder_id': current_grinder_id,
        'grinder_tip': ({'x': int(tip[0]), 'y': int(tip[1])} if tip else None),
        'stored': tip is not None,
        'all_tips': {str(k): {'x': int(v[0]), 'y': int(v[1])} for k, v in stored_grinder_tips.items()},
        'mapping': CAMERA_SERIALS,
    })


if __name__=='__main__':
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)   # stop logging every request

    print("="*70)
    print("🤖 BLADE GRINDER CONTROL SYSTEM")
    print("   REG 134-136: Detection X/Y/Status")
    print("   REG 137-140: START / GRINDER_READY / GRIND_START / E-STOP")
    print("   REG 141:     TEETH_INSPECT  (Robot→PC, 1=record ON, 0=record OFF)")
    print("   REG 142:     ROBOT_ANGLE    (Robot→PC, ×10 signed degrees)")
    print("="*70)

    # Warm up Spinnaker at startup so the slow one-time discovery happens
    # BEFORE the operator starts working — camera start/switch is then fast.
    try:
        _get_pyspin_system()
    except Exception as e:
        print(f"⚠ Spinnaker warm-up failed (camera will retry on first start): {e}")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)