# Skeleton Stream Testing & Visualization

This folder contains testing and visualization tools for skeleton stream conversion.

## Purpose

Verify that the skeleton stream conversion (binary → numpy) is working correctly before running the full conversion on all 162 files.

## Files

### Testing
- **[test_stream_conversion.py](test_stream_conversion.py)** - Comprehensive test suite
  - Test 1: Load all 162 skel.stream~ files
  - Test 2: Convert one sample file
  - Test 3: Analyze 350-dim structure (Kinect v2)
- **[TEST_README.md](TEST_README.md)** - Detailed testing documentation
- **[QUICK_TEST.md](QUICK_TEST.md)** - Quick reference guide

### Visualization
- **[visualize_skeleton.ipynb](visualize_skeleton.ipynb)** - Google Colab notebook
  - Matplotlib 3D static visualization
  - Plotly interactive 3D visualization
  - Data quality checks
- **[COLAB_VISUALIZATION_GUIDE.md](COLAB_VISUALIZATION_GUIDE.md)** - Step-by-step usage guide

### Documentation
- **[SKELETON_STREAM_SUMMARY.md](SKELETON_STREAM_SUMMARY.md)** - Complete overview

## Quick Start

### 1. Run Tests
```bash
cd /home/s2020425/NoXiRe/data_preparation/skeleton_testing
python test_stream_conversion.py
```

This will:
- ✓ Verify all 162 stream files are readable
- ✓ Convert one sample file
- ✓ Analyze the Kinect structure
- ✓ Save results to `./test_conversion_output/`

### 2. Visualize (Google Colab)
1. Run the test script above first
2. Upload `visualize_skeleton.ipynb` to [Google Colab](https://colab.research.google.com/)
3. Upload `test_conversion_output/sample_expert_skel.npy`
4. Run all cells
5. See your skeleton visualization!

See [COLAB_VISUALIZATION_GUIDE.md](COLAB_VISUALIZATION_GUIDE.md) for detailed instructions.

## Output Structure

```
skeleton_testing/
├── test_conversion_output/       # Created by test script
│   ├── sample_expert_skel.npy          # Raw converted (T, 350)
│   ├── sample_reshaped_25x14.npy       # Reshaped (T, 25, 14)
│   ├── sample_positions_25x3.npy       # Positions only (T, 25, 3)
│   └── analysis_summary.json           # Statistics
└── [test/visualization files]
```

## What Gets Verified

✅ **Conversion correctness:**
- 162 files detected and readable
- Binary → numpy conversion works
- Data shapes are correct

✅ **Kinect structure:**
- 350 dims = 25 joints × 14 features
- First 3 features = (x, y, z) positions
- Position ranges are reasonable

✅ **Visualization:**
- Skeleton has humanoid shape
- Bones connect logically
- Movement is detected

## Next Steps

After verification:
1. ✓ Conversion verified
2. Run full conversion on all 162 files (use `../format_conversion.py`)
3. Decide on feature extraction (positions only vs full data)
4. Integrate with your pipeline

## Dependencies

**Required:**
- Python 3.8+
- numpy

**Optional (for visualization):**
- matplotlib
- plotly
- jupyter (for running notebook locally)

## Related Files

The main conversion script is in the parent directory:
- **[../format_conversion.py](../format_conversion.py)** - Production conversion script

See parent [README](../STREAM_CONVERSION_README.md) for full conversion workflow.
