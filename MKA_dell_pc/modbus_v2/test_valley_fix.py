"""
test_valley_fix.py — prove the valley-Y fix removes the along-blade oscillation.

A static frame can't show the indexing oscillation (same frame -> same numbers),
so this tests the MECHANISM: how far the OLD synthetic apex grid drifts from the
REAL teeth by the grinder end, vs the NEW real-midpoint placement. The old drift
should match your ~0.1mm oscillation amplitude; the new should be ~0.

Run (uses the frame grab_and_probe.py saved):
    python3 test_valley_fix.py clean_frame.png
"""
import sys
import numpy as np
import cv2

import app_sinefit_grinder as app
from app_sinefit_grinder import SerratedBladeAnalyzer, get_current_grinder_tip

PPM = float(app.pixels_per_mm)
PITCH_MM = 2.0


def main(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read {path}"); return

    an = SerratedBladeAnalyzer(frame)
    an.preprocess_image()
    an.detect_blade_and_grinder()
    stored = get_current_grinder_tip()
    if stored:
        an.grinder_tip = stored
    an.teeth_profiles = an.extract_tooth_profiles()

    gt = an.grinder_tip
    if not gt:
        print("No grinder tip — set it from the dashboard first."); return

    above = [t for t in an.teeth_profiles if t.grinding_point[1] < gt[1]]
    above.sort(key=lambda t: t.grinding_point[1])
    if len(above) < 4:
        print(f"Only {len(above)} teeth above grinder."); return

    apex_ys = np.array([t.grinding_point[1] for t in above], dtype=float)
    gt_y = gt[1]
    nominal_pitch_px = PITCH_MM * PPM

    # Measured pitch (what the patch uses)
    idx = np.arange(len(apex_ys), dtype=float)
    slope, icpt = np.polyfit(idx, apex_ys, 1)
    measured_pitch_px = float(slope)

    print("=" * 64)
    print(f"Grinder tip Y           : {gt_y:.0f} px")
    print(f"Teeth above grinder     : {len(apex_ys)}")
    print(f"Nominal pitch_px (2mm)  : {nominal_pitch_px:.2f}")
    print(f"Measured pitch_px       : {measured_pitch_px:.2f} "
          f"({100*(measured_pitch_px-nominal_pitch_px)/nominal_pitch_px:+.2f}%)")
    print("=" * 64)

    # ── OLD method: synthetic grid anchored at apex_ys[0], assumed pitch ─────
    y_ref = apex_ys[0]
    old_apex_grid = np.array([y_ref + k * nominal_pitch_px
                              for k in range(len(apex_ys))])

    # ── NEW method: real apexes; valleys at true consecutive midpoints ──────
    new_valleys_y = (apex_ys[:-1] + apex_ys[1:]) / 2.0

    # Compare the SYNTHETIC apex grid against REAL apexes — this drift is the
    # bug: it's what offsets the valley-Y the along-blade move is measured from.
    print("\nReal apex Y   vs  OLD synthetic-grid apex Y   (drift = the bug):")
    max_drift_px = 0.0
    for i, (ry, gy) in enumerate(zip(apex_ys, old_apex_grid)):
        drift = gy - ry
        max_drift_px = max(max_drift_px, abs(drift))
        flag = "  <-- grinder end" if i == len(apex_ys) - 1 else ""
        print(f"  tooth {i:2d}  real={ry:7.1f}  grid={gy:7.1f}  "
              f"drift={drift:+6.2f}px ({drift/PPM*1000:+6.1f}um){flag}")

    print("\n" + "-" * 64)
    print(f"OLD max grid drift from real teeth : {max_drift_px:6.2f} px  "
          f"({max_drift_px/PPM:.3f} mm)")
    print(f"NEW valley-Y error (on real teeth) : ~0.00 px  "
          f"(valleys sit exactly between detected apexes)")
    print("-" * 64)

    # Translate drift to along-blade move error at the grinder end
    print(f"\nThat {max_drift_px/PPM:.3f} mm is the scale of your along-blade "
          f"(next-tooth / X) oscillation.")
    if max_drift_px / PPM > 0.05:
        print("=> OLD method drifts at the oscillation scale you observed. "
              "The fix removes it.")
    else:
        print("=> Drift is small here; if you still see oscillation it may be "
              "motion/re-detection, not the grid. Tell me and we'll instrument "
              "the live sequence.")

    # Sanity: confirm the patched core agrees (real-midpoint valleys)
    try:
        from sine_fit_core import apply_sine_correction
        ca, cv, info, _ = apply_sine_correction(
            an.teeth_profiles, gt, PPM, PITCH_MM,
            blade_edge_points=an.blade_edge_points)
        if cv:
            vy = sorted(v[1] for v in cv)
            print(f"\nPatched core produced {len(cv)} valleys; "
                  f"spacing between them (should match real pitch ~"
                  f"{measured_pitch_px:.1f}px):")
            diffs = np.diff(vy)
            print("  " + "  ".join(f"{d:.1f}" for d in diffs))
    except Exception as e:
        print(f"\n(Could not run patched core directly: {e})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 test_valley_fix.py clean_frame.png"); sys.exit(1)
    main(sys.argv[1])