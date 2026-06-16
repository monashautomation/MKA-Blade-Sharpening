#!/usr/bin/env python3
"""
apply_patch.py — apply sine-fit corrections to app.py
======================================================
Run from the same directory as app.py and sine_fit_core.py:

    python3 apply_patch.py

Produces  app_sinefit.py  — the original app.py is NOT modified.
Verify it runs, then rename/replace as needed.
"""

import re, sys, os

SRC = 'app.py'
DST = 'app_sinefit.py'

if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found in current directory"); sys.exit(1)

with open(SRC, 'r') as f:
    src = f.read()

changes = []


# ── PATCH 1 — import sine_fit_core ───────────────────────────────────────────
OLD1 = 'app = Flask(__name__)\nCORS(app)'
NEW1 = 'from sine_fit_core import apply_sine_correction, draw_sine_overlay\n\napp = Flask(__name__)\nCORS(app)'
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1); changes.append('PATCH 1  import sine_fit_core')
else:
    print("WARN PATCH 1: 'app = Flask(...)' line not found; add import manually")


# ── PATCH 2 — add pitch_mm global ────────────────────────────────────────────
OLD2 = "pixels_per_mm           = 76.812"
NEW2 = ("pixels_per_mm           = 76.812\n"
        "pitch_mm                = 0.0    # 0.0 = disabled; set from dashboard")
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1); changes.append('PATCH 2  pitch_mm global')
else:
    print("WARN PATCH 2: pixels_per_mm line not found; add  pitch_mm = 0.0  manually")


# ── PATCH 3 — replace _generate_coordinates ──────────────────────────────────
NEW_GENERATE = '''    def _generate_coordinates(self):
        global pixels_per_mm, pitch_mm
        if len(self.teeth_profiles) < 2:
            return None
        grinder_tip = self.grinder_tip if self.grinder_tip else get_current_grinder_tip()
        if not grinder_tip:
            return None

        # ── Sine-fit path (pitch_mm > 0) ─────────────────────────────────────
        if pitch_mm > 0.0:
            result = _generate_coordinates_sine(self, grinder_tip)
            if result is not None:
                return result
            print("[sine_fit] fit failed or too few teeth — falling back to raw detection")

        # ── Raw path (original logic) ─────────────────────────────────────────
        deduped = _dedupe_clusters(
            self.teeth_profiles,
            y_threshold_mm=1.5,
            grinder_tip=grinder_tip,
        )
        if len(deduped) < 2:
            return None

        closest_valley = None
        min_distance = float(\'inf\')
        for i in range(len(deduped) - 1):
            ct = deduped[i]
            nt = deduped[i + 1]
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

            if move_y_mm > 0.8:
                dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
                if dist < min_distance:
                    min_distance = dist
                    closest_valley = {
                        \'valley_x\': valley_x,
                        \'valley_y\': valley_y,
                        \'move_x_mm\': move_x_mm,
                        \'move_y_mm\': move_y_mm,
                        \'depth_x_mm\': depth_x_mm,
                        \'between_teeth\': f"{ct.tooth_id}-{nt.tooth_id}",
                        \'distance_mm\': dist,
                    }

        if not closest_valley:
            return None

        print(closest_valley)
        return {
            \'valley_id\': closest_valley[\'between_teeth\'],
            \'x_mm\':      round(float(closest_valley[\'move_y_mm\']), 2),
            \'y_mm\':      round(float(closest_valley[\'move_x_mm\']), 2),
            \'depth_x_mm\': round(float(closest_valley[\'depth_x_mm\']), 2),
            \'valley_x_px\': int(closest_valley[\'valley_x\']),
            \'valley_y_px\': int(closest_valley[\'valley_y\']),
            \'grinder_tip_x_px\': int(grinder_tip[0]),
            \'grinder_tip_y_px\': int(grinder_tip[1]),
            \'num_teeth\':  int(len(deduped)),
            \'distance_mm\': round(float(closest_valley[\'distance_mm\']), 2),
            \'status\': 1,
            \'all_valleys\': [],
            \'sine_fit\': False,
        }
'''

pattern = r'( {4}def _generate_coordinates\(self\):.*?\'all_valleys\': \[\],\s*\})'
m = re.search(pattern, src, re.DOTALL)
if m:
    src = src[:m.start()] + NEW_GENERATE + src[m.end():]
    changes.append('PATCH 3  _generate_coordinates replaced')
else:
    print("WARN PATCH 3: could not locate _generate_coordinates via regex; patch manually")


# ── PATCH 4 — inject _generate_coordinates_sine module-level function ─────────
SINE_FUNC = '''

def _generate_coordinates_sine(analyzer, grinder_tip):
    """
    Sine-fit path: uses the datasheet pitch + baseline polynomial to correct
    for blade bend before computing valley/apex coordinates.

    Only teeth ABOVE the grinder tip (unsharpened side) are used for fitting.
    Falls back to raw detection if the fit cannot be completed.
    """
    global pixels_per_mm, pitch_mm

    gt_y = grinder_tip[1]
    gt_x = grinder_tip[0]

    corrected_apexes, corrected_valleys, info, above_profiles = apply_sine_correction(
        analyzer.teeth_profiles,
        grinder_tip,
        pixels_per_mm,
        pitch_mm,
        blade_edge_points=analyzer.blade_edge_points,
        min_teeth_for_fit=2,
        baseline_poly_degree=1,
    )

    if corrected_apexes is None or corrected_valleys is None or len(corrected_apexes) < 2:
        return None

    # Attach to analyzer for overlay rendering
    analyzer._sine_corrected_apexes  = corrected_apexes
    analyzer._sine_corrected_valleys = corrected_valleys
    analyzer._sine_info              = info

    # Pick closest valley that is above (move_y_mm > 0) the grinder
    sorted_valleys = sorted(corrected_valleys, key=lambda v: v[1])
    closest_valley = None
    min_distance   = float(\'inf\')

    for (vx, vy) in sorted_valleys:
        move_x_mm = (gt_x - vx) / pixels_per_mm
        move_y_mm = (gt_y - vy) / pixels_per_mm
        if move_y_mm < 0.5:
            continue
        dist = (move_x_mm ** 2 + move_y_mm ** 2) ** 0.5
        if dist < min_distance:
            min_distance = dist
            depth_x_mm   = abs(info[\'A_mean\']) / pixels_per_mm
            apexes_above  = [a for a in corrected_apexes if a[1] < vy]
            ct_lbl        = len(apexes_above)
            closest_valley = {
                \'valley_x\':      vx,
                \'valley_y\':      vy,
                \'move_x_mm\':     move_x_mm,
                \'move_y_mm\':     move_y_mm,
                \'depth_x_mm\':    depth_x_mm,
                \'between_teeth\': f"{ct_lbl}-{ct_lbl+1}",
                \'distance_mm\':   dist,
            }

    if not closest_valley:
        return None

    num_teeth = len([a for a in corrected_apexes if a[1] < gt_y])

    print(f"[sine_fit] ✓ valley {closest_valley[\'between_teeth\']} "
          f"move=({closest_valley[\'move_y_mm\']:.2f}, {closest_valley[\'move_x_mm\']:.2f})mm "
          f"depth={closest_valley[\'depth_x_mm\']:.2f}mm  "
          f"A={info[\'A_mean\']:.1f}px  n_pairs={info[\'n_pairs\']}")

    return {
        \'valley_id\':      closest_valley[\'between_teeth\'],
        \'x_mm\':           round(float(closest_valley[\'move_y_mm\']),  2),
        \'y_mm\':           round(float(closest_valley[\'move_x_mm\']),  2),
        \'depth_x_mm\':     round(float(closest_valley[\'depth_x_mm\']), 2),
        \'valley_x_px\':    int(closest_valley[\'valley_x\']),
        \'valley_y_px\':    int(closest_valley[\'valley_y\']),
        \'grinder_tip_x_px\': int(grinder_tip[0]),
        \'grinder_tip_y_px\': int(grinder_tip[1]),
        \'num_teeth\':      num_teeth,
        \'distance_mm\':    round(float(closest_valley[\'distance_mm\']), 2),
        \'status\':         1,
        \'all_valleys\':    [],
        \'sine_fit\':       True,
        \'pitch_mm\':       pitch_mm,
    }

'''

INJECT_BEFORE = '# ── Inspection helpers ───────────────────────────────────────────────────────'
if INJECT_BEFORE in src:
    src = src.replace(INJECT_BEFORE, SINE_FUNC + INJECT_BEFORE, 1)
    changes.append('PATCH 4  _generate_coordinates_sine injected')
else:
    print("WARN PATCH 4: Inspection helpers comment not found; insert function manually")


# ── PATCH 5 — /api/detection/pitch route ─────────────────────────────────────
PITCH_ROUTE = """
@app.route('/api/detection/pitch', methods=['POST'])
def set_pitch():
    global pitch_mm
    data = request.json or {}
    val  = data.get('pitch_mm', 0)
    try:
        pitch_mm = max(0.0, float(val))
        msg = (f'Pitch = {pitch_mm:.3f} mm — '
               f"sine-fit {'ENABLED' if pitch_mm > 0 else 'DISABLED'}")
        print(f'[pitch] {msg}')
        return jsonify({'success': True, 'pitch_mm': pitch_mm, 'message': msg})
    except (TypeError, ValueError) as e:
        return jsonify({'success': False, 'message': str(e)}), 400

"""

OLD5 = "@app.route('/api/detection/calibrate',methods=['POST'])"
if OLD5 in src:
    src = src.replace(OLD5, PITCH_ROUTE + OLD5, 1)
    changes.append('PATCH 5  /api/detection/pitch route')
else:
    print("WARN PATCH 5: calibrate route not found")


# ── PATCH 6 — detection/status includes pitch_mm ─────────────────────────────
OLD6 = ("    return jsonify({'enabled':detection_enabled,"
        "'last_result':last_detection_result,'pixels_per_mm':pixels_per_mm})")
NEW6 = ("    return jsonify({'enabled':detection_enabled,"
        "'last_result':last_detection_result,'pixels_per_mm':pixels_per_mm,"
        "'pitch_mm':pitch_mm})")
if OLD6 in src:
    src = src.replace(OLD6, NEW6, 1); changes.append('PATCH 6  detection/status pitch_mm')
else:
    print("WARN PATCH 6: detection/status return not found; add pitch_mm manually")


# ── PATCH 7 — sine overlay in get_teeth_profiles ─────────────────────────────
OLD7 = "        analyzer.draw_grinder_v_overlay(frame_to_process)"
NEW7 = ("        analyzer.draw_grinder_v_overlay(frame_to_process)\n"
        "        if pitch_mm > 0 and stored:\n"
        "            _apx, _vly, _info, _ = apply_sine_correction(\n"
        "                profiles, stored, pixels_per_mm, pitch_mm,\n"
        "                blade_edge_points=analyzer.blade_edge_points)\n"
        "            if _info:\n"
        "                draw_sine_overlay(frame_to_process, _apx, _vly, _info)")
if OLD7 in src:
    src = src.replace(OLD7, NEW7, 1); changes.append('PATCH 7  sine overlay in teeth_profiles')
else:
    print("WARN PATCH 7: draw_grinder_v_overlay call not found")


# ── PATCH 8 — overlay label in draw_detection_overlay ─────────────────────────
OVERLAY_INSERT = (
    "        if detection_result.get('sine_fit'):\n"
    "            p_mm = detection_result.get('pitch_mm', 0)\n"
    "            cv2.putText(overlay, f'SINE-FIT  pitch={p_mm:.2f}mm',\n"
    "                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)\n"
)
OLD8 = "    except Exception as e: print(f\"Overlay error: {e}\")\n    return overlay"
NEW8 = OVERLAY_INSERT + "    except Exception as e: print(f\"Overlay error: {e}\")\n    return overlay"
if OLD8 in src:
    src = src.replace(OLD8, NEW8, 1); changes.append('PATCH 8  sine label in draw_detection_overlay')
else:
    print("WARN PATCH 8: overlay except block not found")


# ── Write output ───────────────────────────────────────────────────────────────
with open(DST, 'w') as f:
    f.write(src)

print(f"\n{'='*60}")
print(f"Applied {len(changes)} of 8 patches:")
for c in changes: print(f"  ✓ {c}")
print(f"\nOutput written to: {DST}")
print("Review, then:  cp app_sinefit.py app.py")
