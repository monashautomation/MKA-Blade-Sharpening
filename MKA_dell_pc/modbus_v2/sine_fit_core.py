"""
sine_fit_core.py — Bent-blade tooth/valley correction for SerratedBladeAnalyzer
================================================================================

THEORY
------
The blade edge x-position is:  x(y) = A·sin(2π·y/pitch_px + φ) + f(y)
where:
  A      = tooth amplitude (half-height of tooth)
  pitch  = known from datasheet → period in pixels = pitch_mm × pixels_per_mm
  φ      = phase offset (unknown)
  f(y)   = slowly-varying BASELINE due to blade curvature/bending

Because pitch is KNOWN, we know that:
  Apex   positions occur at Y values where sin = +1  → x_apex   = A + f(y)
  Valley positions occur at Y values where sin = −1  → x_valley = −A + f(y)

From detected apex Y positions (via existing peak detector) and the known pitch:
  • Valley Y positions = apex_Y + pitch_px/2  (midway between consecutive apexes)
  • Valley X positions are read from the raw blade_edge_points at those Y values

For each apex[k] / valley[k] pair:
  A_k         = (x_apex[k] − x_valley[k]) / 2
  baseline_k  = (x_apex[k] + x_valley[k]) / 2   at y = midpoint

We fit a low-degree polynomial f_fit(y) to (midpoint_y, baseline_x) pairs, then:
  corrected_apex_x(y)   = f_fit(y) + A_mean
  corrected_valley_x(y) = f_fit(y) − A_mean

Only teeth ABOVE the grinder tip (unsharpened side) are used — sharpened teeth
have shorter height and would distort the amplitude estimate.

USAGE
-----
from sine_fit_core import apply_sine_correction, draw_sine_overlay

apexes, valleys, info, above = apply_sine_correction(
    teeth_profiles, grinder_tip, pixels_per_mm, pitch_mm)

if apexes:
    draw_sine_overlay(frame, apexes, valleys, info, pitch_mm * pixels_per_mm)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def _lookup_valley_x(blade_edge_points: np.ndarray, target_y: float, window: int = 4) -> Optional[float]:
    """
    Find the blade-edge X value closest to `target_y` by averaging
    edge-point X values within `window` pixels of target_y.

    Returns None if no edge points are found in the window.
    """
    if blade_edge_points is None or len(blade_edge_points) == 0:
        return None

    ys = blade_edge_points[:, 1]
    mask = np.abs(ys - target_y) <= window
    pts = blade_edge_points[mask]

    if len(pts) == 0:
        # Widen search
        mask2 = np.abs(ys - target_y) <= window * 3
        pts = blade_edge_points[mask2]
        if len(pts) == 0:
            return None

    return float(np.mean(pts[:, 0]))


def apply_sine_correction(
    teeth_profiles,
    grinder_tip: Tuple[int, int],
    pixels_per_mm: float,
    pitch_mm: float,
    blade_edge_points: Optional[np.ndarray] = None,
    min_teeth_for_fit: int = 2,
    baseline_poly_degree: int = 1,
):
    """
    Fit the baseline + amplitude model to above-grinder teeth and return
    corrected apex and valley positions.

    Parameters
    ----------
    teeth_profiles     : list[ToothProfile] — full set from extract_tooth_profiles()
    grinder_tip        : (x, y) pixel position of grinder tip
    pixels_per_mm      : calibration scalar
    pitch_mm           : tooth pitch from datasheet (mm)
    blade_edge_points  : Nx2 array of (x, y) raw blade edge pixels (optional;
                         used to look up valley X from the actual edge signal).
                         If None, valley X is estimated geometrically from ToothProfile.
    min_teeth_for_fit  : minimum above-grinder teeth needed to attempt the fit
    baseline_poly_degree: degree of polynomial used to model blade curvature
                          (1 = linear warp, 2 = curved warp)

    Returns
    -------
    corrected_apexes   : list[(x, y)] corrected apex pixel positions (float)
    corrected_valleys  : list[(x, y)] corrected valley pixel positions (float)
    info               : dict with fit diagnostics, or None on failure
    above_profiles     : list[ToothProfile] — the teeth used for the fit
    """
    if not grinder_tip or not teeth_profiles:
        return None, None, None, []

    pitch_px = pitch_mm * pixels_per_mm
    gt_y = grinder_tip[1]

    # ── 1. Select only ABOVE-GRINDER teeth ───────────────────────────────────
    above = [t for t in teeth_profiles if t.grinding_point[1] < gt_y]
    if len(above) < min_teeth_for_fit:
        print(f"[sine_fit] only {len(above)} teeth above grinder "
              f"(need ≥{min_teeth_for_fit}) — skipping fit")
        return None, None, None, above

    # Sort ascending Y (away from grinder → toward grinder)
    above.sort(key=lambda t: t.grinding_point[1])

    apex_ys = np.array([t.grinding_point[1] for t in above], dtype=float)
    apex_xs = np.array([t.grinding_point[0] for t in above], dtype=float)

    # ── 2. Estimate amplitude A from apex–valley pairs ────────────────────────
    # Valley Y = apex Y + pitch_px/2  (midway between this apex and the next)
    valley_ys_pred = apex_ys + pitch_px / 2.0
    valley_xs_meas = []

    for k, (ay, ax) in enumerate(zip(apex_ys, apex_xs)):
        vy = valley_ys_pred[k]
        vx = None

        # Strategy 1: look up from raw blade_edge_points if provided
        if blade_edge_points is not None and len(blade_edge_points) > 0:
            vx = _lookup_valley_x(blade_edge_points, vy, window=int(pitch_px * 0.15))

        # Strategy 2: interpolate from bottom_valley of this tooth profile
        if vx is None:
            t = above[k]
            bv = t.bottom_valley
            if bv:
                # bottom_valley is the valley just below this apex
                # weight toward it, but adjust for any Y offset
                vx = float(bv[0])

        # Strategy 3: interpolate from next tooth's top_valley if available
        if vx is None and k + 1 < len(above):
            t_next = above[k + 1]
            tv = t_next.top_valley
            if tv:
                vx = float(tv[0])

        if vx is not None:
            valley_xs_meas.append(vx)
        else:
            valley_xs_meas.append(None)

    # Filter to pairs with valid valley X
    valid_pairs = [(apex_ys[k], apex_xs[k], valley_ys_pred[k], valley_xs_meas[k])
                   for k in range(len(apex_ys))
                   if valley_xs_meas[k] is not None]

    if len(valid_pairs) < 1:
        print("[sine_fit] no valid apex-valley pairs found — cannot fit")
        return None, None, None, above

    v_apex_ys  = np.array([p[0] for p in valid_pairs])
    v_apex_xs  = np.array([p[1] for p in valid_pairs])
    v_valley_ys = np.array([p[2] for p in valid_pairs])
    v_valley_xs = np.array([p[3] for p in valid_pairs])

    A_estimates    = (v_apex_xs - v_valley_xs) / 2.0
    midpoint_ys    = (v_apex_ys + v_valley_ys) / 2.0
    baseline_xs    = (v_apex_xs + v_valley_xs) / 2.0

    # Robust amplitude: median of per-pair estimates
    A_mean = float(np.median(A_estimates))

    if abs(A_mean) < 0.5:
        print(f"[sine_fit] amplitude {A_mean:.2f}px suspiciously small — aborting")
        return None, None, None, above

    # ── 3. Fit polynomial baseline f(y) ──────────────────────────────────────
    deg = min(baseline_poly_degree, len(midpoint_ys) - 1)
    deg = max(deg, 0)

    try:
        coeffs = np.polyfit(midpoint_ys, baseline_xs, deg=deg)
        f_fit  = np.poly1d(coeffs)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"[sine_fit] polyfit failed: {e}")
        return None, None, None, above

    # ── 4. Build corrected apex/valley positions for the full Y range ─────────
    # Cover from the topmost above-grinder apex down to (and slightly past) the grinder.
    y_top    = apex_ys[0]
    y_bottom = gt_y + pitch_px * 0.5   # one half-pitch past the grinder

    # Phase: apex Y positions → y_peak_k = y_ref + k*pitch_px where y_ref aligns
    # with the first detected apex.
    y_ref = apex_ys[0]
    k_max = int(np.ceil((y_bottom - y_ref) / pitch_px)) + 2

    corrected_apexes  = []
    corrected_valleys = []

    for k in range(-1, k_max):
        y_apex_k = y_ref + k * pitch_px
        if y_apex_k < y_top - pitch_px:
            continue
        x_apex_k = float(f_fit(y_apex_k)) + A_mean
        corrected_apexes.append((x_apex_k, y_apex_k))

        y_val_k = y_apex_k + pitch_px / 2.0
        x_val_k = float(f_fit(y_val_k)) - A_mean
        corrected_valleys.append((x_val_k, y_val_k))

    info = {
        'A_mean':        A_mean,
        'A_per_pair':    A_estimates.tolist(),
        'poly_degree':   deg,
        'poly_coeffs':   coeffs.tolist(),
        'pitch_px':      pitch_px,
        'n_above':       len(above),
        'n_pairs':       len(valid_pairs),
    }

    print(f"[sine_fit] A={A_mean:.2f}px  f(y)=poly{deg}  "
          f"pairs={len(valid_pairs)}  apexes={len(corrected_apexes)}")

    return corrected_apexes, corrected_valleys, info, above


def draw_sine_overlay(
    img: np.ndarray,
    corrected_apexes,
    corrected_valleys,
    info: dict,
    color_apex=(0, 255, 128),
    color_valley=(255, 128, 0),
    color_baseline=(200, 200, 0),
):
    """
    Draw the corrected sine overlay on a BGR image in-place:
      • smooth baseline curve (yellow)
      • corrected apex markers (green stars)
      • corrected valley markers (orange triangles)
    """
    if info is None or corrected_apexes is None:
        return img

    h, w = img.shape[:2]
    A       = info['A_mean']
    coeffs  = info['poly_coeffs']
    f_fit   = np.poly1d(coeffs)
    pitch_px = info['pitch_px']

    # Compute Y range from corrected positions
    all_ys = [p[1] for p in corrected_apexes] + [p[1] for p in corrected_valleys]
    if not all_ys:
        return img

    y_lo = max(0, int(min(all_ys)) - int(pitch_px))
    y_hi = min(h - 1, int(max(all_ys)) + int(pitch_px))

    # ── Baseline curve (f_fit) ────────────────────────────────────────────────
    pts_base = []
    for y in range(y_lo, y_hi + 1, 2):
        x = int(round(float(f_fit(y))))
        if 0 <= x < w:
            pts_base.append((x, y))
    for i in range(len(pts_base) - 1):
        cv2.line(img, pts_base[i], pts_base[i + 1], color_baseline, 1, cv2.LINE_AA)

    # ── Corrected apex markers ────────────────────────────────────────────────
    for idx, (x, y) in enumerate(corrected_apexes):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.drawMarker(img, (xi, yi), color_apex,
                           markerType=cv2.MARKER_STAR,
                           markerSize=14, thickness=2)
            cv2.putText(img, f"A{idx}", (xi + 8, yi - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_apex, 1)

    # ── Corrected valley markers ──────────────────────────────────────────────
    for idx, (x, y) in enumerate(corrected_valleys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.drawMarker(img, (xi, yi), color_valley,
                           markerType=cv2.MARKER_TRIANGLE_DOWN,
                           markerSize=10, thickness=2)
            cv2.putText(img, f"V{idx}", (xi + 8, yi + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_valley, 1)

    # ── Info text ─────────────────────────────────────────────────────────────
    cv2.putText(img,
                f"SINE-FIT  A={A:.1f}px  poly{info['poly_degree']}  pairs={info['n_pairs']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 2)

    return img
