"""
grinder_pitch_probe.py — diagnose sine-fit phase walk on a clean blade.
Place next to app_sinefit_grinder.py and sine_fit_core.py. Run:
    python3 grinder_pitch_probe.py path/to/clean_frame.png
"""
import sys, numpy as np, cv2
from app_sinefit_grinder import SerratedBladeAnalyzer, get_current_grinder_tip
import app_sinefit_grinder as app

PIXELS_PER_MM = app.pixels_per_mm          # whatever the server is using (76.812)
NOMINAL_PITCH_MM = 2.0

def main(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read {path}"); return

    an = SerratedBladeAnalyzer(frame)
    an.preprocess_image()
    an.detect_blade_and_grinder()
    stored = get_current_grinder_tip()
    if stored: an.grinder_tip = stored
    an.teeth_profiles = an.extract_tooth_profiles()

    gt = an.grinder_tip
    if not gt:
        print("No grinder tip — set it first or hardcode one here."); return

    above = [t for t in an.teeth_profiles if t.grinding_point[1] < gt[1]]
    above.sort(key=lambda t: t.grinding_point[1])
    if len(above) < 4:
        print(f"Only {len(above)} teeth above grinder — need a frame with more."); return

    ys  = np.array([t.grinding_point[1] for t in above], dtype=float)
    idx = np.arange(len(ys), dtype=float)

    # ── Measured period via regression: slope of Y vs tooth index ────────────
    slope, intercept = np.polyfit(idx, ys, 1)        # px per tooth
    fit  = slope * idx + intercept
    resid = ys - fit
    rms   = float(np.sqrt(np.mean(resid**2)))

    measured_pitch_px = slope
    assumed_pitch_px  = NOMINAL_PITCH_MM * PIXELS_PER_MM
    err_pct = 100.0 * (measured_pitch_px - assumed_pitch_px) / assumed_pitch_px

    # implied calibration if datasheet 2.000 mm is trusted
    implied_ppm = measured_pitch_px / NOMINAL_PITCH_MM
    beat_teeth  = abs(1.0 / (err_pct/100.0)) if abs(err_pct) > 1e-6 else float('inf')

    print("="*60)
    print(f"Teeth above grinder used : {len(ys)}")
    print(f"Assumed  pitch_px (2mm)  : {assumed_pitch_px:8.3f}  (px/mm={PIXELS_PER_MM})")
    print(f"MEASURED pitch_px        : {measured_pitch_px:8.3f}")
    print(f"Pitch error              : {err_pct:+.2f} %")
    print(f"Implied beat length      : {beat_teeth:6.1f} teeth")
    print(f"Regression residual RMS  : {rms:6.2f} px"
          f"   ({rms/PIXELS_PER_MM*1000:.1f} µm)")
    print(f"Implied REAL px/mm        : {implied_ppm:8.3f}"
          f"   (current setting {PIXELS_PER_MM})")
    print("="*60)

    # ── Residual shape — tells you WHICH problem you have ────────────────────
    print("\nPer-tooth residual (px), apex Y vs linear fit:")
    for i,(y,r) in enumerate(zip(ys,resid)):
        bar = "#"*int(abs(r)*4)
        print(f"  tooth {i:2d}  y={y:7.1f}  resid={r:+6.2f}  {bar}")
    sign_flips = np.sum(np.diff(np.sign(resid)) != 0)
    print(f"\nResidual sign flips: {sign_flips} / {len(resid)-1}")
    print(" - near-zero, no structure  → CNC-perfect spacing; "
          "beat is calibration (use MEASURED pitch_px).")
    print(" - many flips (period-2)    → alternating geometry (you ruled out).")
    print(" - smooth curvature         → perspective; pitch varies across frame.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 grinder_pitch_probe.py clean_frame.png"); sys.exit(1)
    main(sys.argv[1])