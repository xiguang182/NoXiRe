# Data Preparation Folder Structure

This folder contains data preparation scripts for the NoXiRe dataset.

## Directory Structure

```
data_preparation/
├── data_pkl.py              # Original pickle-based implementation
├── hdf5_pipeline/           # NEW: HDF5-based pipeline (recommended)
│   ├── README.md           # Main documentation (START HERE)
│   ├── QUICK_START.md      # Quick reference guide
│   ├── INDEX.md            # Navigation guide
│   │
│   ├── Core Implementation:
│   ├── data_hdf5.py                # OpenFace features only
│   ├── data_hdf5_flexible.py       # OpenFace, ViT, or mixed features
│   ├── extract_vit_features.py     # Extract ViT embeddings from videos
│   │
│   ├── Training & Examples:
│   ├── pytorch_example.py          # PyTorch Dataset & Model examples
│   ├── compare_formats.py          # Pickle vs HDF5 benchmarks
│   ├── normalization_comparison.py # Visualize normalization effects
│   ├── test_installation.py        # Verify dependencies
│   │
│   └── Documentation:
│       ├── NORMALIZATION_GUIDE.md  # Why different normalization?
│       ├── README_HDF5.md          # HDF5 format details
│       └── WORKFLOW.md             # Visual workflows
│
└── README_STRUCTURE.md      # This file
```

## Quick Links

### For New Users
👉 **Start here:** [hdf5_pipeline/README.md](hdf5_pipeline/README.md)

### Original Implementation
- **[data_pkl.py](data_pkl.py)** - Original pickle-based format
  - Uses min-max scaling for OpenFace features
  - Stores data as pickle file
  - Kept for reference/backward compatibility

### New HDF5 Pipeline (Recommended)
- **[hdf5_pipeline/](hdf5_pipeline/)** - New implementation with:
  - HDF5 format (faster, smaller, better slicing)
  - Support for ViT features (no normalization)
  - Support for mixed features
  - Complete documentation

## Key Differences

| Feature | data_pkl.py (Old) | hdf5_pipeline/ (New) |
|---------|-------------------|----------------------|
| Format | Pickle | HDF5 |
| Features | OpenFace only | OpenFace, ViT, or mixed |
| Size | Larger | 40-60% smaller |
| Random access | Slow (load all) | Fast (direct access) |
| Slicing | Load all → slice | Slice directly |
| ViT support | No | Yes (with proper normalization) |

## Migration Guide

### If you're using data_pkl.py:

**Old code:**
```python
import pickle

with open('./data/test.pkl', 'rb') as f:
    data = pickle.load(f)

expert_face = data[0][0]['face']
```

**New code:**
```python
from hdf5_pipeline.data_hdf5 import load_sample

data = load_sample('./data/openface.h5', sample_idx=0, person='expert')
expert_face = data['face']
```

**To convert existing data:**
```python
# Your old data is already saved as pickle
# To convert to HDF5, run:
cd hdf5_pipeline
python data_hdf5.py  # Creates HDF5 file from OpenFace CSVs
```

## Which Should I Use?

### Use `data_pkl.py` if:
- You need backward compatibility
- You already have pickle files
- You prefer the simple pickle format

### Use `hdf5_pipeline/` if:
- Starting a new project ✓
- Working with large datasets ✓
- Using ViT features ✓
- Need efficient slicing ✓
- Want better performance ✓

**Recommendation: Use hdf5_pipeline/ for all new work!**

## Getting Started

1. **Navigate to the new pipeline:**
   ```bash
   cd hdf5_pipeline
   ```

2. **Read the documentation:**
   - Start with [README.md](hdf5_pipeline/README.md)
   - Quick start: [QUICK_START.md](hdf5_pipeline/QUICK_START.md)

3. **Test installation:**
   ```bash
   python test_installation.py
   ```

4. **Run the pipeline:**
   ```bash
   # For OpenFace features
   python data_hdf5.py

   # For ViT features
   python extract_vit_features.py
   python data_hdf5_flexible.py
   ```

## Important Note: ViT Feature Normalization

**Key Question:** Should ViT features be min-max normalized?

**Answer:** NO! Use as-is.

- ViT features are already normalized by LayerNorm
- Min-max scaling distorts semantic relationships
- See [hdf5_pipeline/NORMALIZATION_GUIDE.md](hdf5_pipeline/NORMALIZATION_GUIDE.md) for details

## File Purposes

### Original Files
- `data_pkl.py` - Original implementation (kept for reference)

### HDF5 Pipeline Files

**Core:**
- `data_hdf5.py` - OpenFace features with min-max scaling
- `data_hdf5_flexible.py` - Flexible: OpenFace, ViT, or mixed
- `extract_vit_features.py` - Extract ViT embeddings

**Examples:**
- `pytorch_example.py` - Training examples
- `compare_formats.py` - Performance comparison
- `normalization_comparison.py` - Visualization

**Testing:**
- `test_installation.py` - Verify setup

**Documentation:**
- `README.md` - Main guide
- `QUICK_START.md` - Quick reference
- `NORMALIZATION_GUIDE.md` - Normalization deep dive
- `README_HDF5.md` - HDF5 format details
- `WORKFLOW.md` - Visual workflows
- `INDEX.md` - Navigation guide

## Summary

- **Old approach:** `data_pkl.py` → Pickle format
- **New approach:** `hdf5_pipeline/` → HDF5 format with flexible features
- **Recommendation:** Use HDF5 pipeline for new projects

👉 **Get started:** [hdf5_pipeline/README.md](hdf5_pipeline/README.md)
