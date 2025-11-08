# Quick Test Guide

## Run the Test (One Command)

```bash
cd /home/s2020425/NoXiRe/data_preparation
python test_stream_conversion.py
```

## What Happens

### Test 1: Load All Files (~5 seconds)
```
Found 162 skeleton stream files
  Expert: 81
  Novice: 81

FILE STATISTICS
Total files: 162
Frame count: min/max/mean
File sizes: min/max/mean/total
```

### Test 2: Convert Sample (~2 seconds)
```
Converting: 001_2016-03-17_Paris/expert.skel.stream~
Shape: (26625, 350)
Dtype: float32
Memory: 37.23 MB

✓ Saved to: ./test_conversion_output/
```

### Test 3: Analyze Structure (~3 seconds)
```
ANALYSIS: 25 joints × 14 features

Position ranges:
  X: [-1.5, 1.5] meters
  Y: [-1.0, 2.0] meters  
  Z: [0.5, 4.0] meters

✓ First 3 features are 3D positions (x, y, z)
✓ Features 3-6 might be quaternion orientation

Most active joint: 7 (HandLeft)
Least active joint: 0 (SpineBase)
```

## Output Files

```
./test_conversion_output/
├── sample_expert_skel.npy        # (T, 350)
├── sample_reshaped_25x14.npy     # (T, 25, 14)
├── sample_positions_25x3.npy     # (T, 25, 3)
└── analysis_summary.json         # Metadata
```

## Total Time: ~10 seconds

## What You'll Learn

✓ Total skeleton files in your dataset
✓ Frame counts and sizes
✓ Data structure (25 joints × 14 features)
✓ Position ranges (x, y, z)
✓ Movement patterns
✓ Kinect v2 joint mapping

## If It Fails

**"ModuleNotFoundError: No module named 'numpy'"**
→ Need numpy installed

**"File not found"**
→ Run from `/home/s2020425/NoXiRe/data_preparation/`

**"No files found"**  
→ Check `../data/aria-noxi/` exists

## After Testing

Review the output to decide:
1. Normalization strategy
2. Integration with HDF5 pipeline
3. Feature selection (all 350 or subset?)
4. Next steps

---

**Ready? Run:** `python test_stream_conversion.py`
