# Data Preparation - Folder Organization

## Overview

The `data_preparation/` directory is now organized into **functional** and **testing** areas.

## Directory Structure

```
data_preparation/
│
├── 📄 Functional Files (Root Level)
│   ├── format_conversion.py              # Main skeleton converter
│   ├── data_pkl.py                       # Legacy pickle loader
│   ├── label_pkl.py                      # Label loader
│   ├── README.md                         # Main documentation
│   └── STREAM_CONVERSION_README.md       # Conversion guide
│
├── 📁 hdf5_pipeline/                     # HDF5 conversion pipeline
│   ├── data_hdf5.py                      # OpenFace → HDF5
│   ├── data_hdf5_flexible.py             # Multi-modal HDF5
│   ├── extract_vit_features.py           # ViT feature extraction
│   ├── pytorch_example.py                # Training examples
│   ├── compare_formats.py                # Benchmark tool
│   └── [documentation files]
│
└── 📁 skeleton_testing/                  # Testing & visualization
    ├── test_stream_conversion.py         # Test suite
    ├── visualize_skeleton.ipynb          # Colab notebook
    ├── README.md                         # Testing guide
    ├── TEST_README.md                    # Test details
    ├── COLAB_VISUALIZATION_GUIDE.md      # Visualization guide
    └── [other docs]
```

## Design Principles

### Root Level = Production/Functional
- **What:** Scripts you run for actual data processing
- **Examples:**
  - `format_conversion.py` - Convert all 162 files
  - `data_pkl.py` - Load existing data
  - `label_pkl.py` - Load labels

### Subfolders = Specialized/Development
- **hdf5_pipeline/** - Optional HDF5 integration
- **skeleton_testing/** - Testing and verification

## Usage Patterns

### Production Workflow (Root)
```bash
# Convert skeleton streams
python format_conversion.py

# Load data
python data_pkl.py

# Load labels
python label_pkl.py
```

### Testing Workflow (skeleton_testing/)
```bash
# Run tests
cd skeleton_testing
python test_stream_conversion.py

# Then visualize in Google Colab
# Upload: visualize_skeleton.ipynb + test_conversion_output/sample_expert_skel.npy
```

### Optional HDF5 (hdf5_pipeline/)
```bash
# Convert to HDF5 format
cd hdf5_pipeline
python data_hdf5_flexible.py

# Compare performance
python compare_formats.py

# Extract ViT features
python extract_vit_features.py
```

## File Categories

### Core Production Scripts
| File | Purpose | Location |
|------|---------|----------|
| format_conversion.py | Skeleton stream converter | Root |
| data_pkl.py | Data loader | Root |
| label_pkl.py | Label loader | Root |

### Testing & Verification
| File | Purpose | Location |
|------|---------|----------|
| test_stream_conversion.py | Test suite | skeleton_testing/ |
| visualize_skeleton.ipynb | Visualization | skeleton_testing/ |

### Optional Tools
| File | Purpose | Location |
|------|---------|----------|
| data_hdf5*.py | HDF5 conversion | hdf5_pipeline/ |
| extract_vit_features.py | ViT extraction | hdf5_pipeline/ |
| pytorch_example.py | Training examples | hdf5_pipeline/ |

## Documentation Map

### Getting Started
1. **[README.md](README.md)** - Start here for overview

### Skeleton Conversion
1. **[STREAM_CONVERSION_README.md](STREAM_CONVERSION_README.md)** - Main guide
2. **[skeleton_testing/README.md](skeleton_testing/README.md)** - Testing guide
3. **[skeleton_testing/COLAB_VISUALIZATION_GUIDE.md](skeleton_testing/COLAB_VISUALIZATION_GUIDE.md)** - Visualization

### HDF5 Pipeline
1. **[hdf5_pipeline/README.md](hdf5_pipeline/README.md)** - Overview
2. **[hdf5_pipeline/QUICK_START.md](hdf5_pipeline/QUICK_START.md)** - Quick start
3. **[hdf5_pipeline/NORMALIZATION_GUIDE.md](hdf5_pipeline/NORMALIZATION_GUIDE.md)** - Normalization

## Migration from Old Structure

### What Moved
```
Before:                           After:

test_stream_conversion.py    →   skeleton_testing/test_stream_conversion.py
visualize_skeleton.ipynb      →   skeleton_testing/visualize_skeleton.ipynb
TEST_README.md                →   skeleton_testing/TEST_README.md
COLAB_VISUALIZATION_GUIDE.md  →   skeleton_testing/COLAB_VISUALIZATION_GUIDE.md
SKELETON_STREAM_SUMMARY.md    →   skeleton_testing/SKELETON_STREAM_SUMMARY.md
QUICK_TEST.md                 →   skeleton_testing/QUICK_TEST.md
```

### What Stayed (Functional)
```
format_conversion.py          →   Root (main converter)
data_pkl.py                   →   Root (data loader)
label_pkl.py                  →   Root (label loader)
STREAM_CONVERSION_README.md   →   Root (main guide)
```

### Path Updates
All moved files have been updated with correct paths:
- `../data/aria-noxi` → `../../data/aria-noxi`
- `from format_conversion import` → `sys.path` adjustment

## Quick Reference

### I want to...

**Convert all skeleton files:**
```bash
python format_conversion.py
```

**Test conversion first:**
```bash
cd skeleton_testing && python test_stream_conversion.py
```

**Visualize one sample:**
```
Upload skeleton_testing/visualize_skeleton.ipynb to Colab
```

**Use HDF5 format:**
```bash
cd hdf5_pipeline
# See hdf5_pipeline/QUICK_START.md
```

**Load existing data:**
```python
from data_pkl import DataPkl
data = DataPkl(...)
```

## Benefits of This Organization

✅ **Clear separation** - Production vs Testing vs Optional
✅ **Root level is clean** - Only functional scripts
✅ **Easy navigation** - Related files grouped together
✅ **Scalable** - Add new pipelines as subfolders
✅ **Documentation** - README in each folder

---

**Last Updated:** Organization completed with skeleton_testing/ folder creation
