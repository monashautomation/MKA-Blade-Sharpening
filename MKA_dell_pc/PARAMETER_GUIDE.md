# Quick Start Guide - Detecting All Grooves

## 🔍 Problem: Not detecting all grooves?

If your blade has **3-4 grooves** but the system only detects **2**, you need to adjust the detection parameters.

## ⚙️ Solution: Adjust Parameters in `run_analysis.py`

Open `run_analysis.py` and modify these values:

```python
# ==================== CONFIGURATION ====================
WINDOW_SIZE = 50      # ← ADJUST THIS
MIN_DEPTH_PX = 200    # ← ADJUST THIS
# =======================================================
```

## 📊 Parameter Guide

### WINDOW_SIZE (Detection Sensitivity)
Controls how sensitive the detection is to finding grooves:

- **30-40**: High sensitivity - detects more grooves (may include false positives)
- **50-60**: Medium sensitivity - balanced (recommended starting point)
- **70-80**: Low sensitivity - only detects very pronounced grooves

**If missing grooves:** Decrease WINDOW_SIZE (try 30, 35, 40)
**If detecting false grooves:** Increase WINDOW_SIZE (try 60, 70)

### MIN_DEPTH_PX (Noise Filtering)
Filters out shallow grooves that might be noise:

- **150-200**: Allows shallower grooves
- **200-250**: Medium filtering (recommended)
- **300+**: Only deep grooves

**If missing real grooves:** Decrease MIN_DEPTH_PX (try 150, 180)
**If detecting noise:** Increase MIN_DEPTH_PX (try 250, 300)

## 🧪 Finding Optimal Parameters

### Method 1: Use the Tuning Tool

Run the parameter tuning helper:
```bash
python tune_parameters.py your_image.png
```

This tests multiple parameter combinations and shows which works best.

### Method 2: Manual Testing

1. Start with `WINDOW_SIZE=50, MIN_DEPTH_PX=200`
2. Run the analysis
3. Count detected grooves
4. Adjust:
   - Too few grooves? → Decrease WINDOW_SIZE by 10
   - Too many grooves? → Increase WINDOW_SIZE by 10
   - Detecting noise at image edges? → Increase MIN_DEPTH_PX by 50
5. Repeat until correct

## 📝 Example: Detecting 3 Grooves Instead of 2

If your blade has 3 grooves but only 2 are detected:

```python
# Try these settings:
WINDOW_SIZE = 35      # More sensitive
MIN_DEPTH_PX = 180    # Allow slightly shallower grooves
```

Run again and check the output. Keep adjusting until you see:
```
✓ Detected 3 grooves to grind
```

## ✅ Verification Checklist

After adjusting parameters, verify:

1. **Correct count**: Number of detected grooves matches your blade
2. **Valid positions**: Groove X-coordinates are on the blade (not at image edge like x=15)
3. **Reasonable depths**: Groove depths are similar (e.g., all 250-300px)
4. **Visualization**: Check `blade_analysis.png` - magenta circles should be in groove centers

## 🎯 Your Current Detection

For the image you uploaded:
- **Detected**: 2 grooves
- **Positions**: (310, 324) and (342, 676)
- **Depths**: 298px and 261px
- **Grinder tip**: (1003, 518)

If you have more grooves than this, try:
```python
WINDOW_SIZE = 35
MIN_DEPTH_PX = 150
```

## 🔧 Advanced: Understanding Detection

The algorithm works by:
1. Scanning the blade edge (left side of image)
2. Finding peaks (tooth tips pointing right)
3. Finding valleys (grooves between teeth)
4. Filtering valleys by depth to remove noise

Edge grooves (top/bottom of blade) are harder to detect because they don't have peaks on both sides. The updated code handles this, but requires good parameter tuning.

## 📞 Still Having Issues?

Check these common problems:

**Problem**: All grooves detected at x=15 or similar
- **Cause**: Detecting image boundary, not blade
- **Fix**: Check your image - blade should be clearly visible on left side

**Problem**: No grooves detected
- **Cause**: Parameters too strict
- **Fix**: Try WINDOW_SIZE=30, MIN_DEPTH_PX=100

**Problem**: 10+ grooves detected
- **Cause**: Detecting noise/small variations
- **Fix**: Try WINDOW_SIZE=70, MIN_DEPTH_PX=300

Good luck! 🔪✨
