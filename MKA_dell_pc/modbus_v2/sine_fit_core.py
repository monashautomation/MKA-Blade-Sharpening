# """
# sine_fit_core.py — Bent-blade tooth/valley correction for SerratedBladeAnalyzer
# ================================================================================
#
# THEORY
# ------
# The blade edge x-position is:  x(y) = A·sin(2π·y/pitch_px + φ) + f(y)
# where:
#   A      = tooth amplitude (half-height of tooth)
#   pitch  = known from datasheet → period in pixels = pitch_mm × pixels_per_mm
#   φ      = phase offset (unknown)
#   f(y)   = slowly-varying BASELINE due to blade curvature/bending
#
# All tooth apexes sit at the sine peak (sin = +1), so:
#   apex_x = A + f(y)   →  the envelope f_envelope(y) = f(y) + A
#
# Valley X is the geometric midpoint between two adjacent envelope values:
#   valley_x = (f_envelope(y_apex_k) + f_envelope(y_apex_{k+1})) / 2
#   No edge signal lookup — immune to cracks and burrs in the valley.
#
# Teeth being actively ground are shorter than full-height teeth and are
# excluded from the envelope fit to avoid pulling the polynomial inward
# at the grinder end.
# """
#
# import numpy as np
# import cv2
# from typing import List, Tuple, Optional
#
#
# def _find_confirmed_apex_x(
#     blade_edge_points: np.ndarray,
#     target_y: float,
#     pitch_px: float,
#     search_frac: float = 0.35,
# ) -> Optional[float]:
#     """
#     Find the true apex X by searching for the maximum X in blade_edge_points
#     within a window of ±search_frac*pitch_px around the predicted apex Y.
#
#     Takes the median of the top 20% of X values to reject isolated burr spikes
#     while still finding the real tooth tip (which is consistently far right).
#
#     Returns None if fewer than 3 edge points are found in the window.
#     """
#     if blade_edge_points is None or len(blade_edge_points) == 0:
#         return None
#
#     window = pitch_px * search_frac
#     ys = blade_edge_points[:, 1]
#     mask = np.abs(ys - target_y) <= window
#     pts = blade_edge_points[mask]
#
#     if len(pts) < 3:
#         # Widen search once before giving up
#         mask2 = np.abs(ys - target_y) <= window * 2
#         pts = blade_edge_points[mask2]
#         if len(pts) < 3:
#             return None
#
#     xs = np.sort(pts[:, 0])[::-1]          # descending
#     top_n = max(1, len(xs) // 5)           # top 20%
#     return float(np.median(xs[:top_n]))
#
#
# def apply_sine_correction(
#     teeth_profiles,
#     grinder_tip: Tuple[int, int],
#     pixels_per_mm: float,
#     pitch_mm: float,
#     blade_edge_points: Optional[np.ndarray] = None,
#     min_teeth_for_fit: int = 2,
#     baseline_poly_degree: int = 1,
# ):
#     """
#     Fit a polynomial envelope to the confirmed apex X positions of
#     above-grinder, full-height teeth, then compute corrected apex and
#     valley positions geometrically.
#
#     Parameters
#     ----------
#     teeth_profiles     : list[ToothProfile]
#     grinder_tip        : (x, y) pixel position of grinder tip
#     pixels_per_mm      : calibration scalar
#     pitch_mm           : tooth pitch from datasheet (mm)
#     blade_edge_points  : Nx2 array of (x, y) raw blade edge pixels
#     min_teeth_for_fit  : minimum above-grinder teeth needed
#     baseline_poly_degree : polynomial degree for envelope fit (1 = linear)
#
#     Returns
#     -------
#     corrected_apexes   : list[(x, y)] corrected apex pixel positions
#     corrected_valleys  : list[(x, y)] corrected valley pixel positions
#     info               : dict with fit diagnostics
#     above_profiles     : list[ToothProfile] — teeth above grinder
#     """
#     if not grinder_tip or not teeth_profiles:
#         return None, None, None, []
#
#     pitch_px = pitch_mm * pixels_per_mm
#     gt_y = grinder_tip[1]
#
#     # ── 1. Select only ABOVE-GRINDER teeth ───────────────────────────────────
#     above = [t for t in teeth_profiles if t.grinding_point[1] < gt_y]
#     if len(above) < min_teeth_for_fit:
#         print(f"[sine_fit] only {len(above)} teeth above grinder "
#               f"(need ≥{min_teeth_for_fit}) — skipping fit")
#         return None, None, None, above
#
#     above.sort(key=lambda t: t.grinding_point[1])
#
#     # ── 2. Get apex Y positions from peak detector ────────────────────────────
#     apex_ys = np.array([t.grinding_point[1] for t in above], dtype=float)
#
#     # ── 3. Confirm apex X by searching the actual edge signal ─────────────────
#     # Median of top 20% X values in ±35% pitch window → robust to burr spikes
#     apex_xs = []
#     for i, ay in enumerate(apex_ys):
#         confirmed_x = _find_confirmed_apex_x(blade_edge_points, ay, pitch_px)
#         if confirmed_x is None:
#             confirmed_x = float(above[i].grinding_point[0])
#         apex_xs.append(confirmed_x)
#     apex_xs = np.array(apex_xs, dtype=float)
#
#     # ── 4. Filter to full-height teeth for envelope fit ───────────────────────
#     # Teeth being actively ground are shorter than full-height teeth.
#     # Including them pulls the polynomial inward at the grinder end,
#     # causing the extrapolated valley X to be too shallow.
#     heights = np.array([t.height for t in above], dtype=float)
#
#     valid_heights = heights[heights > 1.0]
#     if len(valid_heights) == 0:
#         return None, None, None, above
#
#     full_height_median = float(np.median(valid_heights))
#     height_threshold   = full_height_median * 0.85
#     full_mask          = heights >= height_threshold
#
#     if full_mask.sum() < 1:
#         # Nothing passes — use all teeth as fallback
#         full_mask = np.ones(len(heights), dtype=bool)
#
#     fit_ys = apex_ys[full_mask]
#     fit_xs = apex_xs[full_mask]
#     A_mean = float(np.median(heights[full_mask]))
#
#     print(f"[sine_fit] {full_mask.sum()}/{len(above)} teeth used for fit "
#           f"(height ≥{height_threshold:.1f}px), A={A_mean:.1f}px")
#
#     # ── 5. Fit polynomial envelope f_envelope(y) = f(y) + A ──────────────────
#     deg = min(baseline_poly_degree, len(fit_ys) - 1)
#     deg = max(deg, 0)
#     try:
#         coeffs     = np.polyfit(fit_ys, fit_xs, deg=deg)
#         f_envelope = np.poly1d(coeffs)
#     except (np.linalg.LinAlgError, ValueError) as e:
#         print(f"[sine_fit] polyfit failed: {e}")
#         return None, None, None, above
#
#     # ── 6. Generate corrected apex and valley positions ───────────────────────
#     # Apex  Y : raw detected Y (reliable from peak detector)
#     # Apex  X : f_envelope(apex_y)   — bend-corrected tip position
#     # Valley Y: apex_y + pitch_px/2  — midway between consecutive apexes
#     # Valley X: (f_envelope(apex_y) + f_envelope(apex_y + pitch_px)) / 2
#     #           Pure geometry — no edge lookup, immune to cracks and burrs
#
#     y_ref    = apex_ys[0]
#     y_bottom = gt_y + pitch_px * 0.5   # one half-pitch past the grinder
#     k_max    = int(np.ceil((y_bottom - y_ref) / pitch_px)) + 2
#
#     corrected_apexes  = []
#     corrected_valleys = []
#
#     for k in range(-1, k_max):
#         y_apex = y_ref + k * pitch_px
#         if y_apex < y_ref - pitch_px:
#             continue
#         x_apex      = float(f_envelope(y_apex))
#         y_next_apex = y_apex + pitch_px
#         x_next_apex = float(f_envelope(y_next_apex))
#
#         corrected_apexes.append((x_apex, y_apex))
#
#         y_val = y_apex + pitch_px / 2.0
#         x_val = (x_apex + x_next_apex) / 2.0   # geometric midpoint
#         corrected_valleys.append((x_val, y_val))
#
#     info = {
#         'A_mean':        A_mean,
#         'poly_degree':   deg,
#         'poly_coeffs':   coeffs.tolist(),
#         'pitch_px':      pitch_px,
#         'n_above':       len(above),
#         'n_full':        int(full_mask.sum()),
#         'height_thresh': height_threshold,
#     }
#
#     print(f"[sine_fit] envelope=poly{deg}  "
#           f"apexes={len(corrected_apexes)}  valleys={len(corrected_valleys)}")
#
#     return corrected_apexes, corrected_valleys, info, above
#
#
# def draw_sine_overlay(
#     img: np.ndarray,
#     corrected_apexes,
#     corrected_valleys,
#     info: dict,
#     color_apex=(0, 255, 128),
#     color_valley=(255, 128, 0),
#     color_baseline=(200, 200, 0),
# ):
#     """
#     Draw the corrected sine overlay on a BGR image in-place:
#       • smooth envelope curve (yellow)
#       • corrected apex markers (green stars)
#       • corrected valley markers (orange triangles)
#     """
#     if info is None or corrected_apexes is None:
#         return img
#
#     h, w = img.shape[:2]
#     coeffs     = info['poly_coeffs']
#     f_envelope = np.poly1d(coeffs)
#     pitch_px   = info['pitch_px']
#     A_mean     = info['A_mean']
#
#     all_ys = [p[1] for p in corrected_apexes] + [p[1] for p in corrected_valleys]
#     if not all_ys:
#         return img
#
#     y_lo = max(0, int(min(all_ys)) - int(pitch_px))
#     y_hi = min(h - 1, int(max(all_ys)) + int(pitch_px))
#
#     # Envelope curve
#     pts_env = []
#     for y in range(y_lo, y_hi + 1, 2):
#         x = int(round(float(f_envelope(y))))
#         if 0 <= x < w:
#             pts_env.append((x, y))
#     for i in range(len(pts_env) - 1):
#         cv2.line(img, pts_env[i], pts_env[i + 1], color_baseline, 1, cv2.LINE_AA)
#
#     # Corrected apex markers
#     for idx, (x, y) in enumerate(corrected_apexes):
#         xi, yi = int(round(x)), int(round(y))
#         if 0 <= xi < w and 0 <= yi < h:
#             cv2.drawMarker(img, (xi, yi), color_apex,
#                            markerType=cv2.MARKER_STAR,
#                            markerSize=14, thickness=2)
#             cv2.putText(img, f"A{idx}", (xi + 8, yi - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_apex, 1)
#
#     # Corrected valley markers
#     for idx, (x, y) in enumerate(corrected_valleys):
#         xi, yi = int(round(x)), int(round(y))
#         if 0 <= xi < w and 0 <= yi < h:
#             cv2.drawMarker(img, (xi, yi), color_valley,
#                            markerType=cv2.MARKER_TRIANGLE_DOWN,
#                            markerSize=10, thickness=2)
#             cv2.putText(img, f"V{idx}", (xi + 8, yi + 12),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_valley, 1)
#
#     # Info text
#     n_full  = info.get('n_full', '?')
#     n_above = info.get('n_above', '?')
#     cv2.putText(img,
#                 f"SINE-FIT  A={A_mean:.1f}px  poly{info['poly_degree']}"
#                 f"  teeth={n_full}/{n_above}",
#                 (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 2)
#
#     return img

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

All tooth apexes sit at the sine peak (sin = +1), so:
  apex_x = A + f(y)   →  the envelope f_envelope(y) = f(y) + A

Valley X is the geometric midpoint between two adjacent envelope values:
  valley_x = (f_envelope(y_apex_k) + f_envelope(y_apex_{k+1})) / 2
  No edge signal lookup — immune to cracks and burrs in the valley.

Teeth being actively ground are shorter than full-height teeth and are
excluded from the envelope fit to avoid pulling the polynomial inward
at the grinder end.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def _find_confirmed_apex_x(
    blade_edge_points: np.ndarray,
    target_y: float,
    pitch_px: float,
    search_frac: float = 0.35,
) -> Optional[float]:
    """
    Find the true apex X by searching for the maximum X in blade_edge_points
    within a window of ±search_frac*pitch_px around the predicted apex Y.

    Takes the median of the top 20% of X values to reject isolated burr spikes
    while still finding the real tooth tip (which is consistently far right).

    Returns None if fewer than 3 edge points are found in the window.
    """
    if blade_edge_points is None or len(blade_edge_points) == 0:
        return None

    window = pitch_px * search_frac
    ys = blade_edge_points[:, 1]
    mask = np.abs(ys - target_y) <= window
    pts = blade_edge_points[mask]

    if len(pts) < 3:
        # Widen search once before giving up
        mask2 = np.abs(ys - target_y) <= window * 2
        pts = blade_edge_points[mask2]
        if len(pts) < 3:
            return None

    xs = np.sort(pts[:, 0])[::-1]          # descending
    top_n = max(1, len(xs) // 5)           # top 20%
    return float(np.median(xs[:top_n]))


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
    Fit a polynomial envelope to the confirmed apex X positions of
    above-grinder, full-height teeth, then compute corrected apex and
    valley positions geometrically.

    Parameters
    ----------
    teeth_profiles     : list[ToothProfile]
    grinder_tip        : (x, y) pixel position of grinder tip
    pixels_per_mm      : calibration scalar
    pitch_mm           : tooth pitch from datasheet (mm)
    blade_edge_points  : Nx2 array of (x, y) raw blade edge pixels
    min_teeth_for_fit  : minimum above-grinder teeth needed
    baseline_poly_degree : polynomial degree for envelope fit (1 = linear)

    Returns
    -------
    corrected_apexes   : list[(x, y)] corrected apex pixel positions
    corrected_valleys  : list[(x, y)] corrected valley pixel positions
    info               : dict with fit diagnostics
    above_profiles     : list[ToothProfile] — teeth above grinder
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

    above.sort(key=lambda t: t.grinding_point[1])

    # ── 2. Get apex Y positions from peak detector ────────────────────────────
    apex_ys = np.array([t.grinding_point[1] for t in above], dtype=float)

    # ── 3. Confirm apex X by searching the actual edge signal ─────────────────
    # Median of top 20% X values in ±35% pitch window → robust to burr spikes
    apex_xs = []
    for i, ay in enumerate(apex_ys):
        confirmed_x = _find_confirmed_apex_x(blade_edge_points, ay, pitch_px)
        if confirmed_x is None:
            confirmed_x = float(above[i].grinding_point[0])
        apex_xs.append(confirmed_x)
    apex_xs = np.array(apex_xs, dtype=float)

    # ── 4. Filter to full-height teeth for envelope fit ───────────────────────
    # Teeth being actively ground are shorter than full-height teeth.
    # Including them pulls the polynomial inward at the grinder end,
    # causing the extrapolated valley X to be too shallow.
    heights = np.array([t.height for t in above], dtype=float)

    valid_heights = heights[heights > 1.0]
    if len(valid_heights) == 0:
        return None, None, None, above

    full_height_median = float(np.median(valid_heights))
    height_threshold   = full_height_median * 0.85
    full_mask          = heights >= height_threshold

    if full_mask.sum() < 1:
        # Nothing passes — use all teeth as fallback
        full_mask = np.ones(len(heights), dtype=bool)

    fit_ys = apex_ys[full_mask]
    fit_xs = apex_xs[full_mask]
    A_mean = float(np.median(heights[full_mask]))

    print(f"[sine_fit] {full_mask.sum()}/{len(above)} teeth used for fit "
          f"(height ≥{height_threshold:.1f}px), A={A_mean:.1f}px")

    # ── 5. Fit polynomial envelope f_envelope(y) = f(y) + A ──────────────────
    deg = min(baseline_poly_degree, len(fit_ys) - 1)
    deg = max(deg, 0)
    try:
        coeffs     = np.polyfit(fit_ys, fit_xs, deg=deg)
        f_envelope = np.poly1d(coeffs)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"[sine_fit] polyfit failed: {e}")
        return None, None, None, above

    # ── 5b. Measured period (px) from REAL apex spacing ──────────────────────
    # The detected apex Ys are CNC-clean (verified ~30µm RMS, <1% pitch error).
    # Use their true spacing for the single below-grinder extrapolation rather
    # than the datasheet pitch. Guarded: a tooth miscount would throw the slope
    # off by ~P/(N-1); the ±12% / RMS gates reject it and fall back to nominal
    # so a corrupted slope can never poison the result.
    measured_pitch_px = pitch_px
    if len(apex_ys) >= 4:
        _idx = np.arange(len(apex_ys), dtype=float)
        _slope, _icpt = np.polyfit(_idx, apex_ys, 1)
        _rms = float(np.sqrt(np.mean((apex_ys - (_slope * _idx + _icpt)) ** 2)))
        if abs(_slope - pitch_px) <= 0.12 * pitch_px and _rms < 0.25 * pitch_px:
            measured_pitch_px = float(_slope)
            print(f"[sine_fit] measured pitch_px={measured_pitch_px:.2f} "
                  f"(nominal {pitch_px:.2f}, "
                  f"{100*(measured_pitch_px-pitch_px)/pitch_px:+.1f}%, "
                  f"rms={_rms:.2f}px)")
        else:
            print(f"[sine_fit] rejected measured pitch (slope={_slope:.2f}, "
                  f"rms={_rms:.2f}) — using nominal {pitch_px:.2f}")

    # ── 6. Corrected apex + valley positions, ANCHORED TO REAL TEETH ─────────
    # Apex  X : f_envelope(real_apex_y)  — bend-corrected, X-denoised
    # Apex  Y : real detected apex Y     — clean from the peak detector
    # Valley  : TRUE midpoint between two CONSECUTIVE REAL apexes.
    #
    # The previous version rebuilt a SYNTHETIC uniform grid from a single
    # anchor (apex_ys[0]) marched by the assumed pitch_px. With a 0.9% pitch
    # error that grid drifted ~9px (~0.12mm) by the grinder end, and the
    # along-blade move oscillated as the drifting grid slid past the grinder.
    # Anchoring to real teeth removes that entirely; pitch is now used ONLY for
    # the one extrapolated valley below the last tooth (a single half-pitch,
    # so a 1% error is sub-pixel).
    corrected_apexes = [(float(f_envelope(ay)), float(ay)) for ay in apex_ys]

    corrected_valleys = []
    for i in range(len(apex_ys) - 1):
        y0, y1 = float(apex_ys[i]), float(apex_ys[i + 1])
        x0, x1 = float(f_envelope(y0)), float(f_envelope(y1))
        corrected_valleys.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))

    # Extrapolate ONE valley past the last detected tooth toward the grinder,
    # only if the grinder sits beyond it.
    last_y = float(apex_ys[-1])
    if gt_y > last_y:
        y_next = last_y + measured_pitch_px
        y_val  = (last_y + y_next) / 2.0
        x_val  = (float(f_envelope(last_y)) + float(f_envelope(y_next))) / 2.0
        corrected_valleys.append((x_val, y_val))

    info = {
        'A_mean':           A_mean,
        'poly_degree':      deg,
        'poly_coeffs':      coeffs.tolist(),
        'pitch_px':         measured_pitch_px,   # the value actually used
        'nominal_pitch_px': pitch_px,
        'n_above':          len(above),
        'n_full':           int(full_mask.sum()),
        'height_thresh':    height_threshold,
    }

    print(f"[sine_fit] envelope=poly{deg}  "
          f"apexes={len(corrected_apexes)}  valleys={len(corrected_valleys)}")

    return corrected_apexes, corrected_valleys, info, above


def draw_sine_overlay(
    img: np.ndarray,
    corrected_apexes,
    corrected_valleys,
    info: dict,
    color_apex=(0, 255, 128),
    color_valley=(255, 128, 0),
    color_envelope=(120, 120, 0),
    color_sine=(0, 255, 255),
    sine_sign=-1.0,
    draw_envelope=True,
):
    """
    Draw the sine-fit overlay on a BGR image in-place.

    Unlike the previous version (dots on a straight envelope line), this draws
    the reconstructed tooth-edge SINE the model represents:

        x(y) = f_envelope(y) + sine_sign * (A/2) * (1 - cos(2*pi*(y - y0)/pitch))

    so the curve peaks on the tip line at every real apex Y and dips one full
    tooth-height (A_mean) toward the blade body at each valley.

    All sizes scale with image width, so the overlay stays legible after the
    feed is downscaled into the dashboard panel.

    Notes
    -----
    * Only above-grinder teeth are fitted (see apply_sine_correction), so the
      curve intentionally spans only that region and stops at the grind point.
    * If the wave bulges AWAY from the blade body, flip sine_sign to +1.0.
    * Peak-to-trough is set to A_mean (the median tooth height the fit computed);
      scale (A/2) here if your ToothProfile.height is a half-height.
    """
    if info is None or corrected_apexes is None:
        return img

    h, w = img.shape[:2]
    coeffs     = info.get('poly_coeffs')
    if coeffs is None:
        return img
    f_envelope = np.poly1d(coeffs)
    pitch_px   = float(info.get('pitch_px') or 0.0)
    A_mean     = float(info.get('A_mean') or 0.0)

    all_ys = [p[1] for p in corrected_apexes] + [p[1] for p in (corrected_valleys or [])]
    if not all_ys or pitch_px <= 1.0:
        return img

    # ── Size everything as a fraction of the frame so it survives downscaling ─
    s       = max(1.0, w / 640.0)
    th_sine = max(2, int(round(3.0 * s)))
    th_env  = max(1, int(round(1.0 * s)))
    th_mark = max(2, int(round(2.0 * s)))
    sz_apex = max(12, int(round(11.0 * s)))
    sz_val  = max(10, int(round(9.0  * s)))
    fs_lbl  = max(0.45, 0.42 * s)
    fs_hdr  = max(0.55, 0.55 * s)
    th_txt  = max(1, int(round(1.4 * s)))
    dx      = int(round(8.0 * s))

    y_lo = max(0, int(min(all_ys)) - int(pitch_px))
    y_hi = min(h - 1, int(max(all_ys)) + int(pitch_px))

    # ── Envelope / tip line (dim reference) ──────────────────────────────────
    if draw_envelope:
        prev = None
        for y in range(y_lo, y_hi + 1, 2):
            x = int(round(float(f_envelope(y))))
            if 0 <= x < w:
                cur = (x, y)
                if prev is not None:
                    cv2.line(img, prev, cur, color_envelope, th_env, cv2.LINE_AA)
                prev = cur
            else:
                prev = None

    # ── Reconstructed sine edge curve ────────────────────────────────────────
    # Apexes sit at the sine peak; phase anchored to the first real apex Y and
    # the measured pitch, so peaks line up with the green apex markers.
    apex_ys = sorted(p[1] for p in corrected_apexes)
    y0      = float(apex_ys[0])
    amp     = A_mean / 2.0   # zero-to-peak; peak-to-trough = A_mean = tooth height
    prev = None
    for y in range(y_lo, y_hi + 1):
        phase = 2.0 * np.pi * (y - y0) / pitch_px
        x = float(f_envelope(y)) + sine_sign * amp * (1.0 - np.cos(phase))
        xi = int(round(x))
        if 0 <= xi < w:
            cur = (xi, y)
            if prev is not None:
                cv2.line(img, prev, cur, color_sine, th_sine, cv2.LINE_AA)
            prev = cur
        else:
            prev = None

    # ── Corrected apex markers (green stars) ─────────────────────────────────
    for idx, (x, y) in enumerate(corrected_apexes):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.drawMarker(img, (xi, yi), color_apex,
                           markerType=cv2.MARKER_STAR,
                           markerSize=sz_apex, thickness=th_mark)
            cv2.putText(img, f"A{idx}", (xi + dx, yi - dx),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_lbl, color_apex, th_txt, cv2.LINE_AA)

    # ── Corrected valley markers (orange triangles) ──────────────────────────
    for idx, (x, y) in enumerate(corrected_valleys or []):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.drawMarker(img, (xi, yi), color_valley,
                           markerType=cv2.MARKER_TRIANGLE_DOWN,
                           markerSize=sz_val, thickness=th_mark)
            cv2.putText(img, f"V{idx}", (xi + dx, yi + int(1.5 * dx)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_lbl, color_valley, th_txt, cv2.LINE_AA)

    # ── Info header (black outline for legibility over the bright blade) ──────
    n_full  = info.get('n_full', '?')
    n_above = info.get('n_above', '?')
    hdr = (f"SINE-FIT  A={A_mean:.1f}px  poly{info.get('poly_degree', '?')}"
           f"  teeth={n_full}/{n_above}")
    org = (int(10 * s), int(24 * s))
    cv2.putText(img, hdr, org, cv2.FONT_HERSHEY_SIMPLEX, fs_hdr, (0, 0, 0), th_txt + 3, cv2.LINE_AA)
    cv2.putText(img, hdr, org, cv2.FONT_HERSHEY_SIMPLEX, fs_hdr, (0, 255, 128), th_txt, cv2.LINE_AA)

    return img
