# 📚 Complete Documentation Index

Quick navigation to all documentation and code files.

## 🚀 Start Here

**New to this pipeline?** Read in this order:

1. **[README.md](README.md)** - Overview and your question answered
2. **[QUICK_START.md](QUICK_START.md)** - Get coding in 5 minutes
3. **[WORKFLOW.md](WORKFLOW.md)** - Visual guide to the pipeline

## 📖 Documentation Files

### Getting Started
- **[README.md](README.md)** - Main documentation, feature comparison, usage examples
- **[QUICK_START.md](QUICK_START.md)** - TL;DR with code snippets for common scenarios
- **[WORKFLOW.md](WORKFLOW.md)** - Visual workflows and decision trees

### Deep Dives
- **[NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)** - Why OpenFace needs min-max but ViT doesn't
- **[README_HDF5.md](README_HDF5.md)** - HDF5 format details, best practices, troubleshooting

### This File
- **[INDEX.md](INDEX.md)** - You are here! Navigation guide

## 💻 Code Files

### Core Implementation
| File | Purpose | When to Use |
|------|---------|-------------|
| [data_hdf5.py](data_hdf5.py) | Original OpenFace pipeline | OpenFace features only |
| [data_hdf5_flexible.py](data_hdf5_flexible.py) | Flexible pipeline | ViT or mixed features |

### Feature Extraction
| File | Purpose | When to Use |
|------|---------|-------------|
| [extract_vit_features.py](extract_vit_features.py) | Extract ViT embeddings | When using ViT features |

### Examples & Comparison
| File | Purpose | When to Use |
|------|---------|-------------|
| [pytorch_example.py](pytorch_example.py) | PyTorch Dataset/Model examples | Setting up training |
| [compare_formats.py](compare_formats.py) | Pickle vs HDF5 benchmark | Understanding performance |
| [normalization_comparison.py](normalization_comparison.py) | Visualize normalization | Understanding why |

### Testing
| File | Purpose | When to Use |
|------|---------|-------------|
| [test_installation.py](test_installation.py) | Verify setup | Before starting |

### Original
| File | Purpose | When to Use |
|------|---------|-------------|
| [data_pkl.py](data_pkl.py) | Original pickle implementation | Reference only |

## 🎯 Find What You Need

### "I want to..."

#### ...understand why ViT features shouldn't be normalized
→ [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)
→ [normalization_comparison.py](normalization_comparison.py)

#### ...start using HDF5 format right now
→ [QUICK_START.md](QUICK_START.md)
→ [data_hdf5.py](data_hdf5.py) or [data_hdf5_flexible.py](data_hdf5_flexible.py)

#### ...extract ViT features from videos
→ [extract_vit_features.py](extract_vit_features.py)

#### ...train a PyTorch model
→ [pytorch_example.py](pytorch_example.py)

#### ...compare pickle and HDF5 performance
→ [compare_formats.py](compare_formats.py)

#### ...see visual workflows
→ [WORKFLOW.md](WORKFLOW.md)

#### ...understand HDF5 format details
→ [README_HDF5.md](README_HDF5.md)

#### ...verify my installation
→ [test_installation.py](test_installation.py)

## 📊 Cheat Sheets

### Quick Decision Chart
```
Do you have ViT features?
├─ No → Use data_hdf5.py
└─ Yes → Use data_hdf5_flexible.py with vit_normalization='none'
```

### Normalization Quick Reference
```python
# OpenFace
features = (features - min) / (max - min)  # Min-max [0, 1] ✓

# ViT
features = vit_output  # Use as-is ✓
```

### File Format Quick Reference
```python
# Pickle (old way)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)  # Load ALL
    sample = data[0][0]['face']

# HDF5 (new way)
with h5py.File('data.h5', 'r') as hf:
    sample = hf['sample_000']['expert']['face'][:]  # Load ONLY this
```

## 🔍 Search by Topic

### OpenFace Features
- Overview: [README.md](README.md) - "Comparison: OpenFace vs ViT Features"
- Pipeline: [data_hdf5.py](data_hdf5.py)
- Normalization: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - "OpenFace Features"
- Training: [pytorch_example.py](pytorch_example.py) - `OpenFaceModel`

### ViT Features
- Overview: [README.md](README.md) - "Your Question Answered"
- Extraction: [extract_vit_features.py](extract_vit_features.py)
- Pipeline: [data_hdf5_flexible.py](data_hdf5_flexible.py)
- Normalization: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - "ViT Latent Features"
- Training: [pytorch_example.py](pytorch_example.py) - `ViTModel`

### Mixed Features
- Overview: [README.md](README.md) - "Option C: Mixed"
- Pipeline: [data_hdf5_flexible.py](data_hdf5_flexible.py)
- Architecture: [pytorch_example.py](pytorch_example.py) - `MixedModel`
- Workflow: [WORKFLOW.md](WORKFLOW.md) - "Workflow 3"

### HDF5 Format
- Overview: [README.md](README.md) - "Performance Benefits"
- Details: [README_HDF5.md](README_HDF5.md)
- Examples: [README_HDF5.md](README_HDF5.md) - "Usage Examples"
- Best practices: [README_HDF5.md](README_HDF5.md) - "Best Practices"

### Normalization
- Main guide: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)
- Visualization: [normalization_comparison.py](normalization_comparison.py)
- Quick answer: [QUICK_START.md](QUICK_START.md) - "Summary"

### PyTorch Integration
- Dataset: [pytorch_example.py](pytorch_example.py) - `NoXiReHDF5Dataset`
- Models: [pytorch_example.py](pytorch_example.py) - `OpenFaceModel`, `ViTModel`, `MixedModel`
- Training: [pytorch_example.py](pytorch_example.py) - `train_*_model()`

## 📋 By Use Case

### Use Case 1: "I only have OpenFace features"
1. Read: [README.md](README.md) - "Option A: OpenFace Only"
2. Run: [data_hdf5.py](data_hdf5.py)
3. Train: [pytorch_example.py](pytorch_example.py) - `train_openface_model()`

### Use Case 2: "I want to use ViT features"
1. Read: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - "ViT Latent Features"
2. Extract: [extract_vit_features.py](extract_vit_features.py)
3. Save: [data_hdf5_flexible.py](data_hdf5_flexible.py) with `vit_normalization='none'`
4. Train: [pytorch_example.py](pytorch_example.py) - `train_vit_model()`

### Use Case 3: "I want to use both OpenFace and ViT"
1. Read: [WORKFLOW.md](WORKFLOW.md) - "Workflow 3: Mixed Features"
2. Extract ViT: [extract_vit_features.py](extract_vit_features.py)
3. Save mixed: [data_hdf5_flexible.py](data_hdf5_flexible.py) with `feature_type='mixed'`
4. Train: [pytorch_example.py](pytorch_example.py) - `train_mixed_model()`

### Use Case 4: "I want to understand the normalization decision"
1. Read: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)
2. Visualize: [normalization_comparison.py](normalization_comparison.py)
3. Test: [pytorch_example.py](pytorch_example.py) - `compare_training_stability()`

### Use Case 5: "I want to see if HDF5 is worth it"
1. Read: [README.md](README.md) - "Performance Benefits"
2. Run: [compare_formats.py](compare_formats.py)

## 🎓 Learning Path

### Beginner
1. [README.md](README.md) - Get overview
2. [QUICK_START.md](QUICK_START.md) - Run first example
3. [test_installation.py](test_installation.py) - Verify setup

### Intermediate
4. [WORKFLOW.md](WORKFLOW.md) - Understand pipeline
5. [data_hdf5.py](data_hdf5.py) or [data_hdf5_flexible.py](data_hdf5_flexible.py) - Run full pipeline
6. [pytorch_example.py](pytorch_example.py) - Train models

### Advanced
7. [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - Deep understanding
8. [README_HDF5.md](README_HDF5.md) - Optimize HDF5 usage
9. [compare_formats.py](compare_formats.py) - Benchmark

## 🔧 Troubleshooting Guide

### Problem: "I don't know which file to use"
→ [WORKFLOW.md](WORKFLOW.md) - See decision tree
→ [QUICK_START.md](QUICK_START.md) - See use cases

### Problem: "Import errors"
→ [test_installation.py](test_installation.py) - Check dependencies

### Problem: "ViT features look wrong"
→ [extract_vit_features.py](extract_vit_features.py) - `inspect_vit_features()`
→ [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - "Testing Your Choice"

### Problem: "Training is unstable"
→ [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - Check normalization
→ [pytorch_example.py](pytorch_example.py) - `compare_training_stability()`

### Problem: "Out of memory"
→ [README_HDF5.md](README_HDF5.md) - "Memory-Efficient Batch Processing"
→ [pytorch_example.py](pytorch_example.py) - Use `sequence_length` parameter

### Problem: "Slow loading"
→ [compare_formats.py](compare_formats.py) - Check if using HDF5 correctly
→ [README_HDF5.md](README_HDF5.md) - "Best Practices"

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| Should I normalize ViT features? | No! Use as-is. See [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) |
| Which file for OpenFace only? | [data_hdf5.py](data_hdf5.py) |
| Which file for ViT or mixed? | [data_hdf5_flexible.py](data_hdf5_flexible.py) |
| How to extract ViT features? | [extract_vit_features.py](extract_vit_features.py) |
| How to train models? | [pytorch_example.py](pytorch_example.py) |
| Is HDF5 faster than pickle? | Yes! See [compare_formats.py](compare_formats.py) |
| Where's the quick start? | [QUICK_START.md](QUICK_START.md) |

## 📈 File Size Reference

| File Type | Size | Purpose |
|-----------|------|---------|
| Documentation | ||
| README.md | Large | Main documentation |
| QUICK_START.md | Medium | Quick reference |
| NORMALIZATION_GUIDE.md | Large | Deep dive on normalization |
| README_HDF5.md | Large | HDF5 details |
| WORKFLOW.md | Large | Visual workflows |
| INDEX.md | Medium | This file |
| **Code** | ||
| data_hdf5.py | Small | Original pipeline |
| data_hdf5_flexible.py | Medium | Flexible pipeline |
| extract_vit_features.py | Medium | ViT extraction |
| pytorch_example.py | Large | Training examples |
| compare_formats.py | Medium | Benchmarks |
| normalization_comparison.py | Large | Visualizations |
| test_installation.py | Medium | Testing |

## ✅ Checklist

Before you start:
- [ ] Read [README.md](README.md)
- [ ] Run [test_installation.py](test_installation.py)
- [ ] Choose your use case from [QUICK_START.md](QUICK_START.md)

Before training:
- [ ] Understand normalization ([NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md))
- [ ] Create HDF5 file ([data_hdf5.py](data_hdf5.py) or [data_hdf5_flexible.py](data_hdf5_flexible.py))
- [ ] Test loading ([pytorch_example.py](pytorch_example.py))

For ViT features:
- [ ] Extract features ([extract_vit_features.py](extract_vit_features.py))
- [ ] Verify statistics (`inspect_vit_features()`)
- [ ] Use `vit_normalization='none'`

For optimization:
- [ ] Benchmark ([compare_formats.py](compare_formats.py))
- [ ] Review HDF5 best practices ([README_HDF5.md](README_HDF5.md))

## 🎯 The One-Sentence Answer

**OpenFace features need min-max scaling [0,1] because they have different physical units, but ViT features should be used as-is because they're already normalized by LayerNorm and changing them distorts semantic relationships.**

See [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) for the full explanation.

---

## 📚 Summary

- **11 code files** for implementation
- **6 documentation files** for understanding
- **Everything you need** to process NoXiRe data efficiently

**Start here:** [README.md](README.md) → [QUICK_START.md](QUICK_START.md) → [Your chosen pipeline]

Good luck! 🚀
