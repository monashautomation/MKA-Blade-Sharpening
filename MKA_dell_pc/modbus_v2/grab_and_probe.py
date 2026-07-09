"""
grab_and_probe.py — grab a clean frame from the RUNNING Flask server and
diagnose sine-fit phase walk (pitch / calibration beat) on it.

Why over HTTP?
--------------
The Flask process owns the FLIR camera through PySpin. A second process
cannot open the same camera, so we pull the current frame from the server's
/api/camera/frame endpoint instead of touching PySpin ourselves. No
contention, nothing to coordinate.

Prereqs
-------
  • Flask server running (camera LIVE, blade in position).
  • This file sitting next to app_sinefit_grinder.py and sine_fit_core.py.
  • Grinder tip already SET (the probe reads it from grinder_positions.json
    via the imported module — no camera needed for that).

Run
---
    python3 grab_and_probe.py
    python3 grab_and_probe.py --host http://localhost:5000 --frames 5 --save clean_frame.png
"""

import sys
import time
import argparse
import numpy as np
import cv2

try:
    import requests
except ImportError:
    print("Need 'requests' — pip install requests --break-system-packages")
    sys.exit(1)

# Import the running app's analyzer + persisted grinder tip + calibration.
# This executes module-level code (loads json, imports PySpin) but does NOT
# start Flask and does NOT grab the camera.
import app_sinefit_grinder as app
from app_sinefit_grinder import SerratedBladeAnalyzer, get_current_grinder_tip

NOMINAL_PITCH_MM = 2.0


# ── Frame grabbing ────────────────────────────────────────────────────────────
def grab_frame(host, n_frames=5, settle_s=0.15):
    """
    Pull n_frames JPEGs from /api/camera/frame and median-blend them to
    knock down sensor/JPEG noise. Blade is static so blending is safe; it
    only sharpens the tooth-edge statistics. Returns a BGR uint8 image.
    """
    url = f"{host.rstrip('/')}/api/camera/frame"
    frames = []
    for i in range(n_frames):
        try:
            r = requests.get(url, params={'t': int(time.time() * 1000)}, timeout=5)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  frame {i+1}: decode failed, skipping")
                continue
            frames.append(img)
            print(f"  frame {i+1}/{n_frames}: {img.shape[1]}x{img.shape[0]}")
        except Exception as e:
            print(f"  frame {i+1}: grab failed — {e}")
        time.sleep(settle_s)

    if not frames:
        return None

    # Guard against a size mismatch (shouldn't happen on a static feed)
    h, w = frames[0].shape[:2]
    frames = [f for f in frames if f.shape[:2] == (h, w)]
    stack = np.stack(frames, axis=0)
    blended = np.median(stack, axis=0).astype(np.uint8)
    return blended


def check_server(host):
    try:
        r = requests.get(f"{host.rstrip('/')}/api/camera/ready", timeout=5)
        ready = r.json().get('ready', False)
        if not ready:
            print("⚠ Server says camera is NOT ready (not active, or no frame yet).")
            print("  Start the camera from the dashboard, then re-run.")
        return ready
    except Exception as e:
        print(f"✗ Could not reach server at {host}: {e}")
        print("  Is the Flask server running on that host/port?")
        return False


# ── Pitch probe ───────────────────────────────────────────────────────────────
def probe(frame, ppm):
    an = SerratedBladeAnalyzer(frame)
    an.preprocess_image()
    an.detect_blade_and_grinder()

    stored = get_current_grinder_tip()
    if stored:
        an.grinder_tip = stored
    an.teeth_profiles = an.extract_tooth_profiles()

    gt = an.grinder_tip
    if not gt:
        print("\n✗ No grinder tip available (none detected and none stored).")
        print("  Set the grinder tip from the dashboard first.")
        return None

    above = [t for t in an.teeth_profiles if t.grinding_point[1] < gt[1]]
    above.sort(key=lambda t: t.grinding_point[1])
    if len(above) < 4:
        print(f"\n✗ Only {len(above)} teeth above grinder — need ≥4 for a "
              f"meaningful regression. Check framing/detection.")
        return None

    ys  = np.array([t.grinding_point[1] for t in above], dtype=float)
    idx = np.arange(len(ys), dtype=float)

    slope, intercept = np.polyfit(idx, ys, 1)     # px per tooth
    fit   = slope * idx + intercept
    resid = ys - fit
    rms   = float(np.sqrt(np.mean(resid ** 2)))

    measured_pitch_px = float(slope)
    assumed_pitch_px  = NOMINAL_PITCH_MM * ppm
    err_pct = 100.0 * (measured_pitch_px - assumed_pitch_px) / assumed_pitch_px
    implied_ppm = measured_pitch_px / NOMINAL_PITCH_MM
    beat_teeth  = abs(1.0 / (err_pct / 100.0)) if abs(err_pct) > 1e-6 else float('inf')

    print("\n" + "=" * 60)
    print(f"Grinder tip (px)         : {gt}")
    print(f"Teeth above grinder used : {len(ys)}")
    print(f"Assumed  pitch_px (2 mm) : {assumed_pitch_px:8.3f}   (px/mm={ppm})")
    print(f"MEASURED pitch_px        : {measured_pitch_px:8.3f}")
    print(f"Pitch error              : {err_pct:+.2f} %")
    print(f"Implied beat length      : {beat_teeth:6.1f} teeth")
    print(f"Regression residual RMS  : {rms:6.2f} px   "
          f"({rms / ppm * 1000:.1f} µm)")
    print(f"Implied REAL px/mm       : {implied_ppm:8.3f}   "
          f"(current setting {ppm})")
    print("=" * 60)

    print("\nPer-tooth residual (apex Y vs linear fit):")
    for i, (y, r) in enumerate(zip(ys, resid)):
        bar = "#" * int(min(40, abs(r) * 4))
        print(f"  tooth {i:2d}  y={y:7.1f}  resid={r:+6.2f}  {bar}")

    sign_flips = int(np.sum(np.diff(np.sign(resid)) != 0))
    print(f"\nResidual sign flips: {sign_flips} / {len(resid) - 1}")
    print("Interpretation:")
    print(" • RMS tiny, no structure  → CNC-perfect spacing; beat is CALIBRATION.")
    print("                             Use MEASURED pitch_px; implied px/mm is real.")
    print(" • Many flips (period-2)   → alternating geometry (you ruled this out).")
    print(" • Smooth curvature        → perspective; pitch varies across the frame.")

    return {
        'measured_pitch_px': measured_pitch_px,
        'err_pct': err_pct,
        'rms': rms,
        'implied_ppm': implied_ppm,
        'n_teeth': len(ys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='http://localhost:5000',
                    help='Flask server base URL')
    ap.add_argument('--frames', type=int, default=5,
                    help='frames to median-blend for denoise')
    ap.add_argument('--save', default='clean_frame.png',
                    help='where to save the grabbed frame (set "" to skip)')
    args = ap.parse_args()

    ppm = float(app.pixels_per_mm)
    print(f"Calibration in use: {ppm} px/mm")
    print(f"Grabbing from {args.host} …")

    if not check_server(args.host):
        sys.exit(1)

    frame = grab_frame(args.host, n_frames=args.frames)
    if frame is None:
        print("✗ Got no usable frames.")
        sys.exit(1)

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"Saved grabbed frame → {args.save}")

    print("\n⚠ Note: if you JUST ran a detection, the feed may carry overlay")
    print("  markers that pollute edge extraction. If results look odd, clear")
    print("  the detection (re-run with no detection active) and try again.")

    probe(frame, ppm)


if __name__ == "__main__":
    main()
