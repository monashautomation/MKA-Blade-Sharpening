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
modbus_client    = None
client_connected = False

pyspin_system   = None
pyspin_cam      = None
pyspin_cam_list = None
camera_lock     = Lock()
camera_active   = False
camera_thread   = None
last_frame      = None
camera_config   = {'frame_rate': 30.0, 'exposure_time': 10000, 'gain': 0.0}

blade_analyzer        = None
last_detection_result = None
detection_enabled     = False
pixels_per_mm         = 86.96
grinder_position_file = 'grinder_position.json'
stored_grinder_tip    = None

# ── Inspection state ──────────────────────────────────────────────────────────
inspection_active       = False
inspection_thread       = None
last_inspection_summary = None
last_teeth_inspect_reg  = False
last_robot_angle        = 0.0


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
        data = {'grinder_tip': [int(grinder_tip[0]), int(grinder_tip[1])],
                'timestamp':   datetime.now().isoformat()}
        with open(grinder_position_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✓ Saved grinder position: {grinder_tip}")
    except Exception as e:
        print(f"⚠ Could not save grinder position: {e}")


load_grinder_position()

def get_cluster_peak(profiles, start_idx):
    """Collects consecutive close points from start_idx and returns (best_tooth, next_idx)"""
    cluster = [profiles[start_idx]]
    j = start_idx + 1
    while j < len(profiles):
        if abs(cluster[-1].grinding_point[1] - profiles[j].grinding_point[1]) < 1:
            cluster.append(profiles[j])
            j += 1
        else:
            break
    best = max(cluster, key=lambda t: t.grinding_point[0])
    return best, j
# ── Cluster helper ────────────────────────────────────────────────────────────
# def get_cluster_peak(profiles, start_idx, grinder_tip=None):
#     """
#     Select tooth from cluster based on inter-tooth separation:
#
#     - If teeth in cluster are VERY CLOSE (< 1mm apart):
#       Use cluster max (rightmost tooth) — original behavior
#     - If teeth are SPREAD OUT (>= 1mm):
#       Use tooth closest to grinder — new behavior
#
#     Args:
#         profiles: list of ToothProfile objects
#         start_idx: starting index in profiles
#         grinder_tip: (x, y) tuple; if None, uses stored_grinder_tip
#
#     Returns:
#         (best_tooth, next_index)
#     """
#     global pixels_per_mm, stored_grinder_tip
#
#     cluster = [profiles[start_idx]]
#     j = start_idx + 1
#
#     # Build cluster: group teeth with similar Y coordinates (within 2px)
#     while j < len(profiles):
#         if abs(cluster[-1].grinding_point[1] - profiles[j].grinding_point[1]) < 2:
#             cluster.append(profiles[j])
#             j += 1
#         else:
#             break
#
#     # Single tooth in cluster: return it
#     if len(cluster) == 1:
#         return cluster[0], j
#
#     # Multi-tooth cluster: check separation
#     # Calculate Y-separation (in mm) between first and last tooth in cluster
#     first_y = cluster[0].grinding_point[1]
#     last_y = cluster[-1].grinding_point[1]
#     separation_mm = abs(last_y - first_y) / pixels_per_mm
#
#     gt = grinder_tip if grinder_tip else stored_grinder_tip
#
#     if separation_mm < 1.0:
#         # ✓ TIGHT CLUSTER (< 1mm): use rightmost tooth (original behavior)
#         best = max(cluster, key=lambda t: t.grinding_point[0])
#     elif gt:
#         # ✓ SPREAD CLUSTER (>= 1mm) + grinder known: use closest to grinder (new behavior)
#         best = min(cluster, key=lambda t: (
#                                                   (t.grinding_point[0] - gt[0]) ** 2 +
#                                                   (t.grinding_point[1] - gt[1]) ** 2
#                                           ) ** 0.5)
#     else:
#         # ✓ SPREAD CLUSTER but no grinder: fallback to rightmost
#         best = max(cluster, key=lambda t: t.grinding_point[0])
#
#     return best, j


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

    def preprocess_image(self, blur_kernel=3):
        self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
        self.binary  = cv2.adaptiveThreshold(
            self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
        return self.binary

    def detect_blade_and_grinder(self, sampling_step=1):
        blade_edge = []; grinder_points = []
        for y in range(0, self.height, sampling_step):
            row          = self.binary[y, :]
            white_pixels = np.where(row > 150)[0]
            if len(white_pixels) > 5:
                rightmost = white_pixels[white_pixels > self.width // 3 * 2]
                if len(rightmost) > 0:
                    grinder_points.append((rightmost[0], y))
                leftmost_blade = white_pixels[white_pixels < self.width // 2]
                if len(leftmost_blade) > 0:
                    blade_edge.append((leftmost_blade[0], y))
        self.blade_edge_points   = np.array(blade_edge)     if blade_edge     else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None
        if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
            min_x_idx  = np.argmin(self.grinder_edge_points[:, 0])
            self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
            min_x      = self.grinder_edge_points[min_x_idx, 0]
            tip_points = self.grinder_edge_points[np.abs(self.grinder_edge_points[:, 0] - min_x) < 10]
            self.grinder_edge_center = (int(np.mean(tip_points[:, 0])), int(np.mean(tip_points[:, 1])))
        return self.blade_edge_points, self.grinder_tip

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

        if self.image is not None:
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



    def _groove_sharpness_score(self, ct_tip, nt_tip, perp_half=8):
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
        Return pitch_mm, depth_mm and V-groove sharpness for the closest tooth pair.
        sharpness_score : mean perpendicular contrast (0-255) at edge pixels in the groove
                          y-range. Higher = sharper. Sharp ~200+, blurry ~80-130.
        is_sharp        : True if sharpness_score >= sharp_threshold
        """
        global stored_grinder_tip, pixels_per_mm
        grinder_tip = self.grinder_tip if self.grinder_tip else stored_grinder_tip
        if not grinder_tip or len(self.teeth_profiles) < 2: return None
        closest = None; min_dist = float("inf")
        i = 0
        while i < len(self.teeth_profiles) - 1:
            ct, next_i = get_cluster_peak(self.teeth_profiles, i)

            nt, i = get_cluster_peak(self.teeth_profiles, next_i)

            valley_x  = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
            valley_y  = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
            dist      = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest  = {
                    "ct_id":    ct.tooth_id,
                    "nt_id":    nt.tooth_id,
                    "pitch_mm": abs(ct.grinding_point[1] - nt.grinding_point[1]) / pixels_per_mm,
                    "depth_mm": abs(ct.grinding_point[0] - ct.bottom_valley[0])  / pixels_per_mm,
                    "_ct_tip":  ct.grinding_point,
                    "_nt_tip":  nt.grinding_point,
                }
        if not closest:
            return None
        sharpness = self._groove_sharpness_score(
            closest.pop("_ct_tip"),
            closest.pop("_nt_tip"),
        )
        closest["sharpness_score"] = round(sharpness, 2) if sharpness is not None else None
        closest["is_sharp"]        = (sharpness is not None and sharpness >= sharp_threshold)
        return closest

    def analyze_frame(self, use_stored_grinder=True):
        global stored_grinder_tip
        try:
            self.preprocess_image(); self.detect_blade_and_grinder()
            if self.blade_edge_points is None or len(self.blade_edge_points) < 20: return None
            if use_stored_grinder and stored_grinder_tip is not None:
                self.grinder_tip = stored_grinder_tip
            elif self.grinder_tip is not None:
                save_grinder_position(self.grinder_tip)
            self.teeth_profiles = self.extract_tooth_profiles()
            if len(self.teeth_profiles) > 0 and self.grinder_tip:
                return self._generate_coordinates()
            return None
        except Exception as e:
            print(f"Analysis error: {e}"); import traceback; traceback.print_exc(); return None

    def _generate_coordinates(self):
        global stored_grinder_tip, pixels_per_mm
        if len(self.teeth_profiles) < 2: return None
        grinder_tip = self.grinder_tip if self.grinder_tip else stored_grinder_tip
        if not grinder_tip: return None
        closest_valley = None; min_distance = float('inf')
        all_valleys = []
        i = 0
        for i in range(len(self.teeth_profiles) - 1):
            # ct, next_i = get_cluster_peak(self.teeth_profiles, i)
            #
            # if next_i >= len(self.teeth_profiles):
            #     break
            #
            # nt, i = get_cluster_peak(self.teeth_profiles, next_i)
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
        if not closest_valley: return None
        print(closest_valley)
        return {
            'valley_id':        closest_valley['between_teeth'],
            'x_mm':             round(float(closest_valley['move_y_mm']),2),
            'y_mm':             round(float(closest_valley['move_x_mm']),2),
            'valley_x_px':      int(closest_valley['valley_x']),
            'valley_y_px':      int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(grinder_tip[0]),
            'grinder_tip_y_px': int(grinder_tip[1]),
            'num_teeth':        int(len(self.teeth_profiles)),
            'distance_mm':      round(float(closest_valley['distance_mm']),2),
            'status':           1,
            'all_valleys':      all_valleys,
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
    Records (angle, depth_mm, pitch_mm) only when REG 142 changes by ≥ ANGLE_STEP°.
    Polls at ~10 Hz but skips the vision analysis entirely if the angle hasn't moved.
    Stopped by setting inspection_active = False.
    Finalises by finding the sample at maximum depth and storing the result.
    """
    global inspection_active, last_inspection_summary
    global last_frame, stored_grinder_tip, modbus_client, last_robot_angle

    ANGLE_STEP     = 0.1   # minimum angular change (°) before taking a new sample
    SHARP_THRESHOLD = 90.0  # minimum mean contrast (0-255) across edge; sharp~200+, blurry~80-130

    samples            = []
    blurry_samples     = []   # frames that fell below blur threshold (measurements kept)
    event_log          = []   # every event for dashboard + txt file
    last_sampled_angle = None

    def _evt(tag, msg):
        ts   = time.strftime('%H:%M:%S')
        line = f"[{ts}] {tag:5s} {msg}"
        event_log.append(line)
        print(line)

    _evt('START', 'Calibration inspection recording started (angle-change + blur gated)')

    while inspection_active:
        try:
            # ── 1. Read current blade angle from REG 142 ──────────────────────
            angle = last_robot_angle
            if modbus_client and modbus_client.connected:
                val = modbus_client.read_robot_angle()
                if val is not None:
                    angle = val; last_robot_angle = angle

            # ── 2. Skip if angle hasn't changed enough ────────────────────────
            if last_sampled_angle is not None and abs(angle - last_sampled_angle) < ANGLE_STEP:
                time.sleep(0.05)
                continue

            with camera_lock:
                if last_frame is None:
                    time.sleep(0.05); continue
                frame = last_frame.copy()

            analyzer = SerratedBladeAnalyzer(frame)
            analyzer.preprocess_image()
            analyzer.detect_blade_and_grinder()
            if stored_grinder_tip:
                analyzer.grinder_tip = stored_grinder_tip
            analyzer.teeth_profiles = analyzer.extract_tooth_profiles()
            stats = analyzer._get_closest_pair_stats(sharp_threshold=SHARP_THRESHOLD)

            if not stats:
                _evt('SKIP ', 'angle={:.2f}°  — no tooth pair detected'.format(angle))

            elif not stats['is_sharp']:
                last_sampled_angle = angle
                blurry_samples.append({
                    'angle':    round(angle, 2),
                    'depth_mm': round(stats['depth_mm'], 4),
                    'pitch_mm': round(stats['pitch_mm'], 4),
                    'sharpness': stats['sharpness_score'],
                })
                _evt('BLUR ', 'angle={:.2f}°  depth={:.4f}mm  pitch={:.4f}mm  '
                              'blur={}  threshold={}  — below threshold'.format(
                              angle, stats['depth_mm'], stats['pitch_mm'],
                              stats['sharpness_score'], SHARP_THRESHOLD))

            else:
                last_sampled_angle = angle
                samples.append({
                    'angle':    round(angle, 2),
                    'depth_mm': round(stats['depth_mm'], 4),
                    'pitch_mm': round(stats['pitch_mm'], 4),
                    'sharpness': stats['sharpness_score'],
                })
                _evt('OK   ', 'angle={:.2f}°  depth={:.4f}mm  pitch={:.4f}mm  blur={}'.format(
                              angle, stats['depth_mm'], stats['pitch_mm'], stats['sharpness_score']))
                live = samples[-1]
                last_inspection_summary = {
                    'status':        'running',
                    'samples':       len(samples),
                    'skipped_blur':  len(blurry_samples),
                    'live_angle':    live['angle'],
                    'live_depth':    live['depth_mm'],
                    'live_pitch':    live['pitch_mm'],
                    'live_sharpness':     live['sharpness'],
                    'event_log':     list(event_log),
                }

        except Exception as e:
            _evt('ERR  ', 'sample error: {}'.format(e))
        time.sleep(0.1)

    # ── Finalise: report the FIRST sample that crossed below the blur threshold ────
    # This is the transition point — the angle where the groove first went blurry.
    if blurry_samples:
        transition = blurry_samples[0]
        _evt('DONE ', 'first_blur_transition: angle={}°  depth={}mm  pitch={}mm  '
                      'blur={}  sharp_accepted={}  blurry={}'.format(
                      transition['angle'], transition['depth_mm'], transition['pitch_mm'],
                      transition['sharpness'], len(samples), len(blurry_samples)))
        last_inspection_summary = {
            'status':              'complete',
            'samples':             len(samples),
            'skipped_blur':        len(blurry_samples),
            'transition_angle':    transition['angle'],
            'transition_depth_mm': transition['depth_mm'],
            'transition_pitch_mm': transition['pitch_mm'],
            'transition_blur':     transition['sharpness'],
            'event_log':           list(event_log),
            'curve':               samples,
            'blurry_curve':        blurry_samples,
        }
    elif samples:
        # Blade never went blurry — fallback to first accepted sample
        first = samples[0]
        _evt('DONE ', 'no blur transition found — using first sharp sample: '
                      'angle={}°  depth={}mm  pitch={}mm  blur={}'.format(
                      first['angle'], first['depth_mm'], first['pitch_mm'], first['sharpness']))
        last_inspection_summary = {
            'status':              'complete',
            'samples':             len(samples),
            'skipped_blur':        0,
            'transition_angle':    first['angle'],
            'transition_depth_mm': first['depth_mm'],
            'transition_pitch_mm': first['pitch_mm'],
            'transition_blur':     first['sharpness'],
            'note':                'No blur transition detected — showing first sharp sample',
            'event_log':           list(event_log),
            'curve':               samples,
            'blurry_curve':        [],
        }
    else:
        _evt('DONE ', 'no samples at all  blurry={}'.format(len(blurry_samples)))
        last_inspection_summary = {
            'status':        'error',
            'message':       'No samples collected',
            'skipped_blur':  len(blurry_samples),
            'event_log':     list(event_log),
        }

    _write_inspection_log(event_log, samples, blurry_samples)


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

    def write_detection(self, x_mm, y_mm, status):
        if not self.connected: return None
        x_val=int(x_mm*10); y_val=int(y_mm*10)
        x_u16=x_val if x_val>=0 else 65536+x_val
        y_u16=y_val if y_val>=0 else 65536+y_val
        result=self.client.write_registers(address=self.REG_DETECTION_X,values=[x_u16,y_u16,int(status)])
        if not result.isError(): print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm status={status}")
        return result

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
        """Read REG 134–142 (9 registers) in one shot."""
        if not self.connected: return None
        r=self.client.read_holding_registers(address=self.REG_DETECTION_X,count=9)
        if r.isError(): return None
        reg=r.registers
        return {
            'detection_x_mm': self._s16(reg[0])/10.0,
            'detection_y_mm': self._s16(reg[1])/10.0,
            'status':         reg[2],
            'start':          reg[3],
            'grinder_ready':  reg[4],
            'grind_start':    reg[5],
            'estop':          reg[6],
            'teeth_inspect':  bool(reg[7]),
            'robot_angle':    self._s16(reg[8])/10.0,
        }

    def close(self):
        self.client.close(); self.connected=False; print("✓ Modbus connection closed")


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    try:
        with open('robot_control_dashboard.html','r') as f: return f.read()
    except FileNotFoundError:
        import os; return f"<h1>robot_control_dashboard.html not found</h1><p>cwd:{os.getcwd()}</p>",404

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

@app.route('/api/configuration',methods=['POST'])
def send_configuration():
    if not modbus_client or not modbus_client.connected:
        return jsonify({'success':False,'message':'Not connected'}),400
    data=request.json
    try:
        result=modbus_client.write_configuration(
            bay_id=int(data.get('bay_id')),grinder_id=int(data.get('grinder_id')),
            angle=float(data.get('angle')),depth=float(data.get('depth')),
            length=int(data.get('length')),config_version=int(data.get('config_version')))
        if result and not result.isError():
            return jsonify({'success':True,'message':'Configuration sent (REG 128-133)'})
        return jsonify({'success':False,'message':'Failed to send configuration'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

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
            modbus_client.write_detection(x_mm=float(data['x_mm']),y_mm=float(data['y_mm']),status=int(data.get('status',1)))
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

# ── Camera ────────────────────────────────────────────────────────────────────
@app.route('/api/camera/start',methods=['POST'])
def start_camera():
    global pyspin_system,pyspin_cam,pyspin_cam_list,camera_active,camera_thread,last_frame
    try:
        with camera_lock:
            if camera_active: return jsonify({'success':True,'message':'Camera already running'})
            print("🎥 Initializing FLIR camera...")
            pyspin_system=PySpin.System.GetInstance()
            pyspin_cam_list=pyspin_system.GetCameras()
            if pyspin_cam_list.GetSize()==0:
                pyspin_cam_list.Clear(); pyspin_system.ReleaseInstance()
                pyspin_system=pyspin_cam_list=None
                return jsonify({'success':False,'message':'No FLIR cameras detected'}),500
            pyspin_cam=pyspin_cam_list[0]; pyspin_cam.Init()
            _configure_pyspin_camera(); pyspin_cam.BeginAcquisition()
            camera_active=True
            camera_thread=Thread(target=_camera_capture_thread,daemon=True); camera_thread.start()
            print("✓ FLIR camera started")
            return jsonify({'success':True,'message':'FLIR camera started'})
    except PySpin.SpinnakerException as ex: return jsonify({'success':False,'message':f'PySpin error: {ex}'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

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
        node_exp_auto=PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsWritable(node_exp_auto): node_exp_auto.SetIntValue(node_exp_auto.GetEntryByName('Off').GetValue())
        node_exp=PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if PySpin.IsWritable(node_exp): node_exp.SetValue(min(node_exp.GetMax(),camera_config['exposure_time']))
        node_gain_auto=PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsWritable(node_gain_auto): node_gain_auto.SetIntValue(node_gain_auto.GetEntryByName('Off').GetValue())
        node_gain=PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if PySpin.IsWritable(node_gain): node_gain.SetValue(min(node_gain.GetMax(),camera_config['gain']))
    except PySpin.SpinnakerException as ex: print(f"⚠ Camera config warning: {ex}")

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

@app.route('/api/camera/stop',methods=['POST'])
def stop_camera():
    global pyspin_system,pyspin_cam,pyspin_cam_list,camera_active,camera_thread,last_frame
    try:
        camera_active=False
        if camera_thread and camera_thread.is_alive(): camera_thread.join(timeout=2.0)
        with camera_lock:
            if pyspin_cam:
                try: pyspin_cam.EndAcquisition(); pyspin_cam.DeInit()
                except: pass
                pyspin_cam=None
            if pyspin_cam_list: pyspin_cam_list.Clear(); pyspin_cam_list=None
            if pyspin_system: pyspin_system.ReleaseInstance(); pyspin_system=None
            last_frame=None
        return jsonify({'success':True,'message':'Camera stopped'})
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

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
        filename=f'capture_{time.strftime("%Y%m%d_%H%M%S")}.jpg'; cv2.imwrite(filename,frame)
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
        modbus_result=modbus_client.write_detection(x_mm=result['x_mm'],y_mm=result['y_mm'],status=result['status'])
        if modbus_result and not modbus_result.isError():
            return jsonify({'success':True,'detection':result,'message':f"Valley {result.get('valley_id','N/A')} sent"})
        return jsonify({'success':False,'message':'Failed to send to robot'}),500
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/status',methods=['GET'])
def get_detection_status():
    return jsonify({'enabled':detection_enabled,'last_result':last_detection_result,'pixels_per_mm':pixels_per_mm})

@app.route('/api/detection/calibrate',methods=['POST'])
def calibrate_detection():
    global pixels_per_mm
    data=request.json; new_ppm=data.get('pixels_per_mm')
    if new_ppm and new_ppm>0:
        pixels_per_mm=float(new_ppm); return jsonify({'success':True,'pixels_per_mm':pixels_per_mm})
    return jsonify({'success':False,'message':'Invalid value'}),400

@app.route('/api/detection/update_grinder',methods=['POST'])
def update_grinder_position():
    global camera_active,last_frame,stored_grinder_tip
    if not camera_active or last_frame is None:
        return jsonify({'success':False,'message':'Camera not active'}),400
    try:
        with camera_lock: frame_to_process=last_frame.copy()
        analyzer=SerratedBladeAnalyzer(frame_to_process); analyzer.analyze_frame(use_stored_grinder=False)
        if analyzer.grinder_tip:
            stored_grinder_tip=analyzer.grinder_tip; save_grinder_position(stored_grinder_tip)
            return jsonify({'success':True,'grinder_tip':{'x':int(stored_grinder_tip[0]),'y':int(stored_grinder_tip[1])},'message':f'Grinder updated: {stored_grinder_tip}'})
        return jsonify({'success':False,'message':'Could not detect grinder'}),404
    except Exception as e: return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/teeth_profiles',methods=['GET'])
def get_teeth_profiles():
    global camera_active,last_frame,stored_grinder_tip
    try:
        frame_to_process=None
        if camera_active and last_frame is not None:
            with camera_lock: frame_to_process=last_frame.copy()
        if frame_to_process is None:
            grinder={'x':int(stored_grinder_tip[0]),'y':int(stored_grinder_tip[1])} if stored_grinder_tip else None
            return jsonify({'success':True,'teeth':[],'grinder_tip':grinder,'num_teeth':0})
        analyzer=SerratedBladeAnalyzer(frame_to_process)
        analyzer.preprocess_image(); analyzer.detect_blade_and_grinder()
        if stored_grinder_tip: analyzer.grinder_tip=stored_grinder_tip
        profiles=analyzer.extract_tooth_profiles()
        grinder_tip=None
        if analyzer.grinder_tip: grinder_tip={'x':int(analyzer.grinder_tip[0]),'y':int(analyzer.grinder_tip[1])}
        elif stored_grinder_tip: grinder_tip={'x':int(stored_grinder_tip[0]),'y':int(stored_grinder_tip[1])}
        teeth=[{'tooth_id':t.tooth_id,'tip':{'x':int(t.grinding_point[0]),'y':int(t.grinding_point[1])},
                'top_valley':{'x':int(t.top_valley[0]),'y':int(t.top_valley[1])},
                'bottom_valley':{'x':int(t.bottom_valley[0]),'y':int(t.bottom_valley[1])},
                'angle':round(float(t.angle),2),'height':round(float(t.height),2)} for t in profiles]
        return jsonify({'success':True,'teeth':teeth,'grinder_tip':grinder_tip,'num_teeth':len(teeth)})
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/detection/grinder_status',methods=['GET'])
def get_grinder_status():
    if stored_grinder_tip:
        return jsonify({'success':True,'grinder_tip':{'x':int(stored_grinder_tip[0]),'y':int(stored_grinder_tip[1])},'stored':True})
    return jsonify({'success':True,'grinder_tip':None,'stored':False})


if __name__=='__main__':
    print("="*70)
    print("🤖 BLADE GRINDER CONTROL SYSTEM")
    print("   REG 134-136: Detection X/Y/Status")
    print("   REG 137-140: START / GRINDER_READY / GRIND_START / E-STOP")
    print("   REG 141:     TEETH_INSPECT  (Robot→PC, 1=record ON, 0=record OFF)")
    print("   REG 142:     ROBOT_ANGLE    (Robot→PC, ×10 signed degrees)")
    print("="*70)
    app.run(debug=True,host='0.0.0.0',port=5000)