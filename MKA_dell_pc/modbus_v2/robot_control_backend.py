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
camera_config   = {'frame_rate': 30.0 }

blade_analyzer        = None
last_detection_result = None
detection_enabled     = False
pixels_per_mm           = 76.812
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


def _dedupe_clusters(profiles, y_threshold_mm=0.2, grinder_tip=None):
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
        self.grad = 0.0
        self.m1 = 0
        self.m2 = 0
    def preprocess_image(self, blur_kernel=3):
            self.blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 0)
            self.binary  = cv2.adaptiveThreshold(
                self.blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)
            return self.binary

    # def detect_blade_and_grinder(self, sampling_step=1):
    #     blade_edge = []; grinder_points = []
    #     for y in range(0, self.height, sampling_step):
    #         row = self.binary[y, :]
    #         white_pixels = np.where(row > 180)[0]
    #         if len(white_pixels) > 5:
    #             rightmost = white_pixels[white_pixels > self.width // 3 * 2]
    #             if len(rightmost) > 0:
    #                 grinder_points.append((rightmost[0], y))
    #             leftmost_blade = white_pixels[white_pixels < self.width // 2]
    #             if len(leftmost_blade) > 0:
    #                 blade_edge.append((leftmost_blade[0], y))
    #     self.blade_edge_points   = np.array(blade_edge)     if blade_edge     else None
    #     self.grinder_edge_points = np.array(grinder_points) if grinder_points else None
    #     if self.grinder_edge_points is not None and len(self.grinder_edge_points) > 0:
    #         min_x_idx  = np.argmin(self.grinder_edge_points[:, 0])
    #         self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
    #         min_x      = self.grinder_edge_points[min_x_idx, 0]
    #         tip_points = self.grinder_edge_points[np.abs(self.grinder_edge_points[:, 0] - min_x) < 10]
    #         self.grinder_edge_center = (int(np.mean(tip_points[:, 0])), int(np.mean(tip_points[:, 1])))
    #     return self.blade_edge_points, self.grinder_tip
    def detect_blade_and_grinder(self, sampling_step=1):
        """Detect blade edge (left) and grinder (right). Fit V-shape to grinder edge
        and compute tip as the intersection of two fitted lines."""
        blade_edge = []
        grinder_points = []

        for y in range(0, self.height, sampling_step):
            row = self.binary[y, :]
            white_pixels = np.where(row > 250)[0]
            if len(white_pixels) > 5:
                rightmost = white_pixels[white_pixels > self.width // 3 * 2]
                if len(rightmost) > 0:
                    # Leftmost pixel of the grinder blob on this row
                    grinder_points.append((rightmost[0], y))
                leftmost_blade = white_pixels[white_pixels < self.width // 3*2]
                if len(leftmost_blade) > 0:
                    blade_edge.append((leftmost_blade[0], y))

        self.blade_edge_points = np.array(blade_edge) if blade_edge else None
        self.grinder_edge_points = np.array(grinder_points) if grinder_points else None

        # ── Fit V-shape to grinder edge to find apex (tip) ────────────────────────
        self.grinder_tip = None
        self.grinder_edge_center = None
        self.grinder_upper_line = None  # (slope, intercept) for debug/overlay
        self.grinder_lower_line = None

        if self.grinder_edge_points is not None and len(self.grinder_edge_points) >= 6:
            tip, upper, lower,self.grad, self.m1, self.m2= self._fit_grinder_v(self.grinder_edge_points)
            if tip is not None:
                self.grinder_tip = tip
                self.grinder_edge_center = tip
                self.grinder_upper_line = upper
                self.grinder_lower_line = lower
            else:
                # Fallback to old method if V-fit fails
                min_x_idx = np.argmin(self.grinder_edge_points[:, 0])
                self.grinder_tip = tuple(self.grinder_edge_points[min_x_idx])
                min_x = self.grinder_edge_points[min_x_idx, 0]
                tip_points = self.grinder_edge_points[
                    np.abs(self.grinder_edge_points[:, 0] - min_x) < 10]
                self.grinder_edge_center = (
                    int(np.mean(tip_points[:, 0])), int(np.mean(tip_points[:, 1])))

        return self.blade_edge_points, self.grinder_tip

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
            return None, None, None

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
            return None, None, None

        # Sanity check: the fitted tip should lie near the observed points
        # (within the y-range and not wildly off in x)
        y_min, y_max = ys.min(), ys.max()
        if not (y_min - 20 <= tip[1] <= y_max + 20):
            return None, None, None

        x_min = xs.min()
        if tip[0] < x_min - 30 or tip[0] > xs.max() + 30:
            return None, None, None

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
            self.teeth_profiles = self.extract_tooth_profiles()
            if len(self.teeth_profiles) > 0 and self.grinder_tip:
                return self._generate_coordinates()
            return None
        except Exception as e:
            print(f"Analysis error: {e}");import traceback;traceback.print_exc();return None

    def _generate_coordinates(self):
        global pixels_per_mm
        if len(self.teeth_profiles) < 2:
            return None
        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        # … rest unchanged
        if not grinder_tip:
            return None

        # ── Step 1: collapse tooth clusters (remove duplicates) ───────────────────
        deduped = _dedupe_clusters(
            self.teeth_profiles,
            y_threshold_mm=1.5,  # adjust: 0.5mm = teeth closer than this are "same tooth"
            grinder_tip=grinder_tip,  # bias selection toward the grinder
        )

        if len(deduped) < 2:
            return None

        # ── Step 2: iterate ALL consecutive pairs in the deduped list ────────────
        closest_valley = None
        min_distance = float('inf')
        for i in range(len(deduped) - 1):
            # when there is less than three teeth detected
            ct = deduped[i]
            nt = deduped[i + 1]

            # X-depth from tip line down into the valley between ct and nt.
            # ct.bottom_valley is the valley below ct in image-Y, i.e. between ct & nt.
            avg_tip_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2.0
            depth_x_mm = abs(avg_tip_x - ct.bottom_valley[0]) / pixels_per_mm

            if len(deduped) < 3:
                valley_x = avg_tip_x
                valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            else:
                if i - 1 >= 0:
                    pt = deduped[i - 1]
                    pt_pitch = (ct.grinding_point[1] - pt.grinding_point[1]) / 2
                    valley_y = ct.grinding_point[1] + pt_pitch
                else:
                    valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
                valley_x = avg_tip_x

            move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm

            if move_y_mm > 1:
                dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
                if dist < min_distance:
                    min_distance = dist
                    closest_valley = {
                        'valley_x': valley_x,
                        'valley_y': valley_y,
                        'move_x_mm': move_x_mm,
                        'move_y_mm': move_y_mm,
                        'depth_x_mm': depth_x_mm,  # ← NEW
                        'between_teeth': f"{ct.tooth_id}-{nt.tooth_id}",
                        'distance_mm': dist,
                    }
            # if len(deduped) < 3:
            #     ct = deduped[i]
            #     nt = deduped[i + 1]
            #     valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
            #     valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            #     move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            #     move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
            #     if move_y_mm > 0.5:
            #         dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            #         if dist < min_distance:
            #             min_distance = dist
            #             closest_valley = {
            #                 'valley_x': valley_x,
            #                 'valley_y': valley_y,
            #                 'move_x_mm': move_x_mm,
            #                 'move_y_mm': move_y_mm,
            #                 'between_teeth': f"{ct.tooth_id}-{nt.tooth_id}",
            #                 'distance_mm': dist,
            #             }
            # else:
            #     ct = deduped[i]
            #     nt = deduped[i + 1]
            #     if i - 1 >= 0:
            #         pt = deduped[i - 1]  # previous (top, unsharpened) tooth
            #         pt_pitch = (ct.grinding_point[1] - pt.grinding_point[1]) / 2
            #         valley_y = ct.grinding_point[1] + pt_pitch  # half-pitch below ct
            #     else:
            #         valley_y = (ct.grinding_point[1] + nt.grinding_point[1]) / 2
            #     valley_x = (ct.grinding_point[0] + nt.grinding_point[0]) / 2
            #     move_x_mm = (grinder_tip[0] - valley_x) / pixels_per_mm
            #     move_y_mm = (grinder_tip[1] - valley_y) / pixels_per_mm
            #     if move_y_mm > 0.5:
            #         dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
            #         if dist < min_distance:
            #             min_distance = dist
            #             closest_valley = {
            #                 'valley_x': valley_x,
            #                 'valley_y': valley_y,
            #                 'move_x_mm': move_x_mm,
            #                 'move_y_mm': move_y_mm,
            #                 'between_teeth': f"{ct.tooth_id}-{nt.tooth_id}",
            #                 'distance_mm': dist,
            #                     }
        if not closest_valley:
            return None

        print(closest_valley)
        # return {
        #     'valley_id': closest_valley['between_teeth'],
        #     'x_mm': round(float(closest_valley['move_y_mm']), 2),
        #     'y_mm': round(float(closest_valley['move_x_mm']), 2),
        #     'valley_x_px': int(closest_valley['valley_x']),
        #     'valley_y_px': int(closest_valley['valley_y']),
        #     'grinder_tip_x_px': int(grinder_tip[0]),
        #     'grinder_tip_y_px': int(grinder_tip[1]),
        #     'num_teeth': int(len(deduped)),  # now reports deduped count
        #     'distance_mm': round(float(closest_valley['distance_mm']), 2),
        #     'status': 1,
        #     'all_valleys': [],
        # }
        return {
            'valley_id': closest_valley['between_teeth'],
            'x_mm': round(float(closest_valley['move_y_mm']), 2),
            'y_mm': round(float(closest_valley['move_x_mm']), 2),
            'depth_x_mm': round(float(closest_valley['depth_x_mm']), 2),  # ← NEW
            'valley_x_px': int(closest_valley['valley_x']),
            'valley_y_px': int(closest_valley['valley_y']),
            'grinder_tip_x_px': int(grinder_tip[0]),
            'grinder_tip_y_px': int(grinder_tip[1]),
            'num_teeth': int(len(deduped)),
            'distance_mm': round(float(closest_valley['distance_mm']), 2),
            'status': 1,
            'all_valleys': [],
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
    REG_DETECTION_DEPTH = 143  # ← NEW: PC→Robot, X-depth of valley between teeth (×100, signed)

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

    # def write_detection(self, x_mm, y_mm, status):
    #     if not self.connected: return None
    #     x_val=int(x_mm*100); y_val=int(y_mm*100)
    #     x_u16=x_val if x_val>=0 else 65536+x_val
    #     y_u16=y_val if y_val>=0 else 65536+y_val
    #     result=self.client.write_registers(address=self.REG_DETECTION_X,values=[x_u16,y_u16,int(status)])
    #     if not result.isError(): print(f"✓ Detection written: X={x_mm:.2f}mm Y={y_mm:.2f}mm status={status}")
    #     return result
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

    # def read_all_status(self):
    #     """Read REG 134–142 (9 registers) in one shot."""
    #     if not self.connected: return None
    #     r=self.client.read_holding_registers(address=self.REG_DETECTION_X,count=9)
    #     if r.isError(): return None
    #     reg=r.registers
    #     return {
    #         'detection_x_mm': self._s16(reg[0])/100.0,
    #         'detection_y_mm': self._s16(reg[1])/100.0,
    #         'status':         reg[2],
    #         'start':          reg[3],
    #         'grinder_ready':  reg[4],
    #         'grind_start':    reg[5],
    #         'estop':          reg[6],
    #         'teeth_inspect':  bool(reg[7]),
    #         'robot_angle':    self._s16(reg[8])/10.0,
    #     }
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
            'detection_depth_mm': self._s16(reg[9]) / 100.0,  # ← NEW
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
        # if 'x_mm' in data and 'y_mm' in data:
        #     modbus_client.write_detection(x_mm=float(data['x_mm']),y_mm=float(data['y_mm']),status=int(data.get('status',1)))
        if 'x_mm' in data and 'y_mm' in data:
            modbus_client.write_detection(
                x_mm=float(data['x_mm']),
                y_mm=float(data['y_mm']),
                status=int(data.get('status', 1)),
                depth_mm=float(data.get('depth_mm', 0.0)),  # ← NEW
            )
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
def _stop_camera_internal():
    """Stop capture, release PySpin. Safe to call when already stopped."""
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame
    camera_active = False
    th = camera_thread
    camera_thread = None
    if th and th.is_alive():
        th.join(timeout=2.0)
    with camera_lock:
        if pyspin_cam:
            try: pyspin_cam.EndAcquisition(); pyspin_cam.DeInit()
            except Exception: pass
            pyspin_cam = None
        if pyspin_cam_list:
            pyspin_cam_list.Clear(); pyspin_cam_list = None
        if pyspin_system:
            pyspin_system.ReleaseInstance(); pyspin_system = None
        last_frame = None


def _start_camera_internal(grinder_id):
    """Initialise and start PySpin camera for the given grinder id. Returns (ok, msg)."""
    global pyspin_system, pyspin_cam, pyspin_cam_list, camera_active, camera_thread, last_frame
    if camera_active:
        return False, 'Camera already running (stop first)'
    serial = CAMERA_SERIALS.get(grinder_id, '')
    print(f"🎥 Initializing FLIR camera for grinder {grinder_id} "
          f"(serial='{serial or 'auto-index ' + str(grinder_id - 1)}')")
    pyspin_system = PySpin.System.GetInstance()
    pyspin_cam_list = pyspin_system.GetCameras()
    n = pyspin_cam_list.GetSize()
    if n == 0:
        pyspin_cam_list.Clear(); pyspin_system.ReleaseInstance()
        pyspin_system = pyspin_cam_list = None
        return False, 'No FLIR cameras detected'

    selected = None
    selected_idx = None
    if serial:
        for i in range(n):
            cam = pyspin_cam_list[i]
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
            pyspin_cam_list.Clear();
            pyspin_system.ReleaseInstance()
            pyspin_system = pyspin_cam_list = None
            return False, f"Camera with serial '{serial}' not found for grinder {grinder_id}"
        selected = pyspin_cam_list[selected_idx]
    else:
        idx = grinder_id - 1
        if not (0 <= idx < n):
            pyspin_cam_list.Clear(); pyspin_system.ReleaseInstance()
            pyspin_system = pyspin_cam_list = None
            return False, f'Camera index {idx} out of range (have {n})'
        selected = pyspin_cam_list[idx]

    pyspin_cam = selected
    pyspin_cam.Init()
    _configure_pyspin_camera()
    pyspin_cam.BeginAcquisition()
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
    """Enumerate detected FLIR cameras. Camera must be stopped to enumerate."""
    if camera_active:
        return jsonify({'success': False, 'message': 'Stop camera first to list devices',
                        'mapping': CAMERA_SERIALS,
                        'current_grinder_id': current_grinder_id}), 400
    system = None
    cl = None
    try:
        system = PySpin.System.GetInstance()
        cl = system.GetCameras()
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
    finally:
        # Always release in the correct order, even on the success path.
        if cl is not None:
            try: cl.Clear()
            except Exception as e: print(f"⚠ cl.Clear() failed: {e}")
        if system is not None:
            try: system.ReleaseInstance()
            except Exception as e: print(f"⚠ system.ReleaseInstance() failed: {e}")

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
        # modbus_result=modbus_client.write_detection(x_mm=result['x_mm'],y_mm=result['y_mm'],status=result['status'])
        modbus_result = modbus_client.write_detection(
            x_mm=result['x_mm'],
            y_mm=result['y_mm'],
            status=result['status'],
            depth_mm=result.get('depth_x_mm', 0.0),  # ← NEW
        )
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
        if stored: analyzer.grinder_tip = stored
        profiles = analyzer.extract_tooth_profiles()
        grinder_tip = None
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
    print("="*70)
    print("🤖 BLADE GRINDER CONTROL SYSTEM")
    print("   REG 134-136: Detection X/Y/Status")
    print("   REG 137-140: START / GRINDER_READY / GRIND_START / E-STOP")
    print("   REG 141:     TEETH_INSPECT  (Robot→PC, 1=record ON, 0=record OFF)")
    print("   REG 142:     ROBOT_ANGLE    (Robot→PC, ×10 signed degrees)")
    print("="*70)
    app.run(debug=True,host='0.0.0.0',port=5000)