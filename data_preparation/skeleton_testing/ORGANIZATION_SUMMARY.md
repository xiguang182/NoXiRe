# Skeleton Testing Folder - Organization Summary

## What Changed

All **testing and visualization** files have been moved from `data_preparation/` root to this `skeleton_testing/` subfolder.

## Files in This Folder

### Testing
- **[test_stream_conversion.py](test_stream_conversion.py)** - Main test suite
  - Updated paths: `../../data/aria-noxi` (two levels up)
  - Updated import: Uses `sys.path` to import `format_conversion` from parent

### Visualization
- **[visualize_skeleton.ipynb](visualize_skeleton.ipynb)** - Google Colab notebook
  - Fixed Multiple Frame Comparison with shared axis limits
  - Ready to upload to Colab

### Documentation
- **[README.md](README.md)** - This folder's guide
- **[TEST_README.md](TEST_README.md)** - Testing details
- **[QUICK_TEST.md](QUICK_TEST.md)** - Quick reference
- **[COLAB_VISUALIZATION_GUIDE.md](COLAB_VISUALIZATION_GUIDE.md)** - Visualization guide
- **[SKELETON_STREAM_SUMMARY.md](SKELETON_STREAM_SUMMARY.md)** - Complete overview
- **[ORGANIZATION_SUMMARY.md](ORGANIZATION_SUMMARY.md)** - This file

## Path Changes

Since files moved one level deeper, paths were updated:

### In test_stream_conversion.py:
```python
# Before (root level):
root_dir = '../data/aria-noxi'
from format_conversion import StreamConverter

# After (in skeleton_testing/):
root_dir = '../../data/aria-noxi'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from format_conversion import StreamConverter
```

### In visualize_skeleton.ipynb:
No path changes needed - loads from `test_conversion_output/` (created in same folder)

## Usage

### From skeleton_testing/ folder:
```bash
cd /home/s2020425/NoXiRe/data_preparation/skeleton_testing
python test_stream_conversion.py
```

### Output location:
```bash
skeleton_testing/test_conversion_output/
├── sample_expert_skel.npy
├── sample_reshaped_25x14.npy
├── sample_positions_25x3.npy
└── analysis_summary.json
```

## Why This Organization?

### Before:
```
data_preparation/
├── format_conversion.py          # Production
├── data_pkl.py                    # Production
├── test_stream_conversion.py     # Testing (mixed with production)
├── visualize_skeleton.ipynb       # Testing (mixed with production)
└── [many other files mixed together]
```
❌ Testing and production files mixed

### After:
```
data_preparation/
├── format_conversion.py          # Production (root)
├── data_pkl.py                    # Production (root)
└── skeleton_testing/              # Testing (isolated)
    ├── test_stream_conversion.py
    ├── visualize_skeleton.ipynb
    └── [all testing files]
```
✅ Clean separation of concerns

## Benefits

1. **Root is cleaner** - Only functional/production scripts
2. **Testing is isolated** - All verification tools in one place
3. **Easier maintenance** - Related files together
4. **Scalable** - Can add more testing tools here

## Related Documentation

### Parent folder:
- **[../README.md](../README.md)** - Main data_preparation overview
- **[../STREAM_CONVERSION_README.md](../STREAM_CONVERSION_README.md)** - Conversion guide
- **[../FOLDER_ORGANIZATION.md](../FOLDER_ORGANIZATION.md)** - Organization details

### Production script:
- **[../format_conversion.py](../format_conversion.py)** - Main converter

## Quick Commands

**Run tests:**
```bash
cd skeleton_testing
python test_stream_conversion.py
```

**Visualize:**
1. Upload [visualize_skeleton.ipynb](visualize_skeleton.ipynb) to Colab
2. Upload `test_conversion_output/sample_expert_skel.npy`
3. Run all cells

**Go back to production:**
```bash
cd ..  # Back to data_preparation/
python format_conversion.py  # Run full conversion
```

---

**Status:** Organization complete ✓
