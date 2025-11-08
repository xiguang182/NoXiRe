# HDF5 Pipeline Folder Structure

## Files in This Folder

### 📖 Start Here (Documentation)
```
README.md              ⭐ Main documentation - read this first
QUICK_START.md         ⚡ Quick reference with code examples
INDEX.md               📚 Navigation guide to all files
```

### 💻 Core Implementation
```
data_hdf5.py           📦 OpenFace features only (min-max scaling)
data_hdf5_flexible.py  📦 OpenFace, ViT, or mixed features (flexible)
extract_vit_features.py 🎥 Extract ViT embeddings from videos
__init__.py            📦 Package initialization
```

### 🎓 Examples & Tools
```
pytorch_example.py         🔥 PyTorch Dataset & Model examples
compare_formats.py         📊 Benchmark: Pickle vs HDF5
normalization_comparison.py 📈 Visualize normalization effects
test_installation.py       ✅ Verify dependencies and setup
```

### 📚 Deep Dive Documentation
```
NORMALIZATION_GUIDE.md  📖 Why OpenFace needs min-max but ViT doesn't
README_HDF5.md          📖 HDF5 format details and best practices
WORKFLOW.md             📖 Visual workflows and decision trees
STRUCTURE.md            📖 This file
```

## File Purpose Summary

| File | Type | Purpose | When to Use |
|------|------|---------|-------------|
| **README.md** | Doc | Main overview, answers your question | First read |
| **QUICK_START.md** | Doc | Quick reference, code snippets | Quick lookup |
| **INDEX.md** | Doc | Navigate to specific topics | Find something |
| **NORMALIZATION_GUIDE.md** | Doc | Why ViT = as-is, OpenFace = min-max | Understand why |
| **README_HDF5.md** | Doc | HDF5 format details | Deep dive |
| **WORKFLOW.md** | Doc | Visual workflows | See process |
| **data_hdf5.py** | Code | OpenFace → HDF5 | OpenFace only |
| **data_hdf5_flexible.py** | Code | Any features → HDF5 | ViT or mixed |
| **extract_vit_features.py** | Code | Video → ViT embeddings | Need ViT features |
| **pytorch_example.py** | Code | Training examples | Start training |
| **compare_formats.py** | Code | Benchmark pickle vs HDF5 | See performance |
| **normalization_comparison.py** | Code | Visualize effects | Understand effects |
| **test_installation.py** | Code | Check setup | Before starting |

## Recommended Reading Order

### For Beginners
1. **README.md** - Get overview
2. **QUICK_START.md** - Run first example
3. **test_installation.py** - Verify setup works

### For Implementation
4. **data_hdf5.py** OR **data_hdf5_flexible.py** - Choose based on features
5. **pytorch_example.py** - Set up training

### For Understanding
6. **NORMALIZATION_GUIDE.md** - Why different normalization?
7. **WORKFLOW.md** - See visual workflows
8. **README_HDF5.md** - Optimize usage

## Quick Decision Guide

```
What do you want to do?
│
├─ Understand normalization → README.md (answer to your question)
├─ Extract ViT features → extract_vit_features.py
├─ Create HDF5 file → data_hdf5.py or data_hdf5_flexible.py
├─ Train models → pytorch_example.py
├─ See performance → compare_formats.py
├─ Verify setup → test_installation.py
└─ Navigate docs → INDEX.md
```

## File Dependencies

```
extract_vit_features.py
    ↓ generates .npy files
data_hdf5_flexible.py
    ↓ creates .h5 file
pytorch_example.py
    ↓ trains model
```

## Import Structure

```python
# As a package
from hdf5_pipeline import save_to_hdf5, load_sample

# Or directly
from hdf5_pipeline.data_hdf5_flexible import save_to_hdf5
from hdf5_pipeline.pytorch_example import NoXiReHDF5Dataset
```

## File Size Reference

| File | Lines | Complexity |
|------|-------|------------|
| data_hdf5.py | ~250 | Simple |
| data_hdf5_flexible.py | ~350 | Medium |
| extract_vit_features.py | ~300 | Medium |
| pytorch_example.py | ~500 | Medium |
| compare_formats.py | ~300 | Simple |
| normalization_comparison.py | ~400 | Medium |
| test_installation.py | ~400 | Simple |

## Key Concepts per File

### data_hdf5.py
- Min-max scaling for OpenFace
- HDF5 creation
- Basic loading functions

### data_hdf5_flexible.py
- Multiple feature types
- Flexible normalization
- Advanced loading (by name, with filters)

### extract_vit_features.py
- ViT model loading
- Frame extraction
- Feature inspection

### pytorch_example.py
- Custom Dataset classes
- Model architectures
- Training loops
- Two-stream fusion

### NORMALIZATION_GUIDE.md
- Why OpenFace needs scaling
- Why ViT doesn't
- Semantic preservation
- Gradient flow

## Parent Directory

This folder is inside:
```
NoXiRe/data_preparation/
├── data_pkl.py              # Original pickle implementation
├── hdf5_pipeline/           # This folder (NEW)
│   └── [all these files]
└── README_STRUCTURE.md      # Overview of organization
```

See [../README_STRUCTURE.md](../README_STRUCTURE.md) for parent directory info.

## Summary

- **14 files total** in this folder
- **7 code files** for implementation
- **7 documentation files** for understanding
- **Everything you need** for efficient HDF5 data pipeline

**Start with README.md!** 🚀
