# Google Colab Visualization Guide

## Quick Start (3 Steps)

### Step 1: Run Test Script Locally
```bash
cd /home/s2020425/NoXiRe/data_preparation
python test_stream_conversion.py
```

This creates `./test_conversion_output/` with converted numpy files.

### Step 2: Upload to Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `visualize_skeleton.ipynb`
3. Upload files from `test_conversion_output/`:
   - `sample_expert_skel.npy` (required)
   - OR `sample_positions_25x3.npy` (if available)

### Step 3: Run All Cells

Click **Runtime → Run all** and see your skeleton!

---

## What You'll Get

### Visualization 1: Matplotlib (Static 3D)
- Clear skeleton structure with bones
- Multiple viewpoints (start, middle, end)
- Joint labels for key points

### Visualization 2: Plotly (Interactive 3D)
- **Rotate** with mouse
- **Zoom** with scroll
- **Pan** with click-drag
- Hover to see joint names and coordinates

### Data Quality Check
- Position ranges verification
- Movement detection
- Kinect coordinate validation
- **Confirmation that conversion worked!**

---

## Expected Output

You should see a **humanoid skeleton** with:

```
     ●  Head
     │
     ●  Neck
    ╱│╲
   ●─●─●  Shoulders + Arms
     │
     ●  Spine
    ╱ ╲
   ●   ●  Hips
   │   │
   ●   ●  Knees
   │   │
   ●   ●  Feet
```

**Position ranges (typical):**
- X: -1.5 to 1.5 meters (left-right)
- Y: -0.5 to 2.0 meters (bottom-top)
- Z: 0.5 to 4.0 meters (depth)

---

## Files Needed

### Minimum (Option A):
- `sample_expert_skel.npy` (raw 350-dim data)

The notebook will automatically reshape it to (T, 25, 3) for positions.

### Better (Option B):
- `sample_positions_25x3.npy` (pre-extracted positions)

Already reshaped, loads faster.

---

## Troubleshooting

### "No such file or directory"
→ Upload the `.npy` file to Colab's file browser (left sidebar)

### "ModuleNotFoundError: plotly"
→ Uncomment this line in the first cell:
```python
!pip install plotly
```

### "Invalid shape"
→ Make sure you're loading the correct file:
- Use `sample_expert_skel.npy` (350 dims)
- OR `sample_positions_25x3.npy` (3 dims)

### Skeleton looks wrong
→ Check the data quality section output
→ Verify position ranges are reasonable
→ Try visualizing a different frame

---

## Alternative: Run Locally (If You Have Display)

If your local machine becomes available:

```bash
cd /home/s2020425/NoXiRe/data_preparation
jupyter notebook visualize_skeleton.ipynb
```

Or convert to Python script:

```bash
jupyter nbconvert --to python visualize_skeleton.ipynb
python visualize_skeleton.py
```

---

## What This Confirms

✅ **Conversion is correct** if you see:
1. Skeleton has humanoid shape
2. 25 joints are present
3. Bones connect logically (head→neck→spine→legs, etc.)
4. Position ranges are reasonable (~0-3 meters)
5. Movement is detected between frames

❌ **Conversion has issues** if:
1. Skeleton looks random/scattered
2. All joints are at origin (0,0,0)
3. Position ranges are extreme (>10 meters)
4. No movement detected
5. Skeleton structure makes no sense

---

## After Verification

Once you confirm the conversion works:

1. ✅ **Full conversion**: Convert all 162 files
   ```python
   from format_conversion import StreamConverter
   converter = StreamConverter()
   converter.convert_all_streams()
   ```

2. ✅ **Integration**: Decide how to use skeleton data
   - Standalone for skeleton-based tasks
   - Combined with OpenFace/ViT features
   - Added to HDF5 pipeline

3. ✅ **Normalization**: Choose strategy
   - Center by torso position?
   - Normalize by person height?
   - Keep as-is?

---

## Quick Test Checklist

- [ ] Upload notebook to Colab
- [ ] Upload `sample_expert_skel.npy`
- [ ] Run all cells
- [ ] See skeleton visualization
- [ ] Check it looks humanoid
- [ ] Verify position ranges
- [ ] Confirm movement detected
- [ ] ✅ Conversion verified!

---

**Ready to visualize!** 🎨

Upload to [Google Colab](https://colab.research.google.com/) and run!
