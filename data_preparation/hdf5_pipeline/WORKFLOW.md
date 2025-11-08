# Data Pipeline Workflow

Visual guide to the complete data processing pipeline.

## Decision Tree: Which Pipeline Should I Use?

```
┌─────────────────────────────────────────┐
│  What features do you want to use?     │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴──────────┬─────────────────┬──────────────┐
    │                    │                 │              │
    ▼                    ▼                 ▼              ▼
┌─────────┐      ┌──────────┐     ┌────────────┐  ┌──────────┐
│OpenFace │      │   ViT    │     │    Both    │  │  Other   │
│  only   │      │   only   │     │  (Mixed)   │  │ features │
└────┬────┘      └─────┬────┘     └──────┬─────┘  └────┬─────┘
     │                 │                  │             │
     │                 │                  │             │
     ▼                 ▼                  ▼             ▼
┌─────────┐      ┌──────────┐     ┌────────────┐  ┌──────────┐
│data_hdf5│      │ Extract  │     │  Extract   │  │ Adapt    │
│  .py    │      │   ViT    │     │    ViT     │  │data_hdf5_│
│         │      │features  │     │  features  │  │flexible  │
│Min-max  │      │          │     │            │  │  .py     │
│ [0,1]   │      │Use as-is │     │Use as-is   │  │          │
└─────────┘      └──────────┘     └────────────┘  └──────────┘
```

## Workflow 1: OpenFace Features (Original)

```
Input: CSV files
    │
    ├─ face landmarks (299:435)    → Min-max [0,1]
    ├─ action units (679:696)       → Divide by 5
    └─ head pose (293:299)          → Min-max [0,1]
    │
    ▼
data_hdf5.py
    │
    ▼
openface_data.h5
    │
    └─ sample_000/expert/
           ├─ face: (T, 68, 2)     [0, 1]
           ├─ aus: (T, 17)         [0, 1]
           └─ head: (T, 6)         [0, 1]
```

**Normalization:** Min-max [0, 1] ✓
**Why:** Different physical units need alignment

## Workflow 2: ViT Features Only

```
Input: Video files (.mp4)
    │
    ▼
extract_vit_features.py
    │
    ├─ Load ViT model (e.g., google/vit-base-patch16-224)
    ├─ Extract frame-by-frame
    └─ Save .npy files → KEEP AS-IS (no normalization!)
    │
    ▼
.npy files: (T, 768)
    │ mean ≈ 0.0
    │ std ≈ 1.0
    │
    ▼
data_hdf5_flexible.py
    │
    ├─ vit_normalization='none'  ← Important!
    │
    ▼
vit_data.h5
    │
    └─ sample_000/expert/
           └─ vit: (T, 768)       [mean≈0, std≈1]
```

**Normalization:** None (as-is) ✓
**Why:** Already normalized by LayerNorm, preserve semantics

## Workflow 3: Mixed Features (OpenFace + ViT)

```
Input: CSV files + Video files
    │
    ├───────────────┬───────────────┐
    │               │               │
    ▼               ▼               │
OpenFace CSVs   Video files        │
    │               │               │
    │               ▼               │
    │        extract_vit_features   │
    │               │               │
    │               ▼               │
    │          .npy files           │
    │           (as-is)             │
    │               │               │
    └───────────────┴───────────────┘
                    │
                    ▼
        data_hdf5_flexible.py
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
    OpenFace              ViT features
    Min-max [0,1]         Keep as-is
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            mixed_data.h5
                    │
    └─ sample_000/expert/
           ├─ face: (T, 68, 2)  [0, 1]
           ├─ aus: (T, 17)      [0, 1]
           ├─ head: (T, 6)      [0, 1]
           └─ vit: (T, 768)     [mean≈0]
```

**Normalization:** Different per feature type ✓
**Why:** Each feature type has its own optimal distribution

## Training Pipeline

### OpenFace Model
```
HDF5 file (openface_data.h5)
    │
    ▼
NoXiReHDF5Dataset
    │ feature_type='openface'
    │ sequence_length=100
    ▼
DataLoader (batch_size=16)
    │
    ▼
Batch: (16, 100, 159)  [0, 1] range
    │
    ▼
OpenFaceModel
    │ LSTM → FC
    ▼
Output: (16, num_classes)
```

### ViT Model
```
HDF5 file (vit_data.h5)
    │
    ▼
NoXiReHDF5Dataset
    │ feature_type='vit'
    │ sequence_length=100
    ▼
DataLoader (batch_size=16)
    │
    ▼
Batch: (16, 100, 768)  [mean≈0]
    │
    ▼
ViTModel
    │ LSTM → FC
    ▼
Output: (16, num_classes)
```

### Mixed Model (Two-Stream)
```
HDF5 file (mixed_data.h5)
    │
    ▼
NoXiReHDF5Dataset
    │ feature_type='mixed'
    │ sequence_length=100
    ▼
DataLoader (batch_size=16)
    │
    ├─────────────┬─────────────┐
    │             │             │
    ▼             ▼             │
OpenFace      ViT             │
(16,100,159)  (16,100,768)    │
 [0, 1]        [mean≈0]       │
    │             │             │
    ▼             ▼             │
OpenFace      ViT             │
 Branch        Branch          │
(Linear)      (Linear)         │
    │             │             │
    └──────┬──────┘             │
           │                    │
           ▼                    │
    Concatenate                 │
      (512)                     │
           │                    │
           ▼                    │
        LSTM                    │
           │                    │
           ▼                    │
     Fusion Layer               │
           │                    │
           ▼                    │
    (16, num_classes)          │
```

## Feature Statistics

### OpenFace (After Min-Max)
```
┌─────────────────┬──────┬──────┬──────┬──────┐
│ Feature         │ Min  │ Max  │ Mean │ Std  │
├─────────────────┼──────┼──────┼──────┼──────┤
│ Face landmarks  │ 0.0  │ 1.0  │ ~0.5 │ ~0.3 │
│ Action Units    │ 0.0  │ 1.0  │ ~0.2 │ ~0.2 │
│ Head pose       │ 0.0  │ 1.0  │ ~0.5 │ ~0.3 │
└─────────────────┴──────┴──────┴──────┴──────┘

Distribution: Bounded [0, 1]
```

### ViT Features (Original)
```
┌─────────────────┬──────┬──────┬──────┬──────┐
│ Feature         │ Min  │ Max  │ Mean │ Std  │
├─────────────────┼──────┼──────┼──────┼──────┤
│ ViT embeddings  │ ~-4  │ ~+4  │ ~0.0 │ ~1.0 │
└─────────────────┴──────┴──────┴──────┴──────┘

Distribution: Approximately Gaussian
```

## File Format Comparison

### Pickle (.pkl)
```
test.pkl
    │
    ├─ List of tuples (expert, novice)
    │     │
    │     ├─ Expert dict: {'face', 'aus', 'head'}
    │     └─ Novice dict: {'face', 'aus', 'head'}
    │
    ├─ Must load entire file
    ├─ No metadata
    └─ Larger file size

Access pattern:
  data = pickle.load(file)        # Load ALL
  sample = data[0][0]['face']     # Then access
```

### HDF5 (.h5)
```
openface_data.h5
    │
    ├─ Attributes (metadata)
    ├─ sample_names dataset
    │
    ├─ sample_000/
    │     ├─ expert/
    │     │     ├─ face: (T, 68, 2)
    │     │     ├─ aus: (T, 17)
    │     │     └─ head: (T, 6)
    │     └─ novice/
    │           └─ ...
    │
    ├─ sample_001/
    │     └─ ...
    │
    ├─ Direct access to any sample
    ├─ Efficient slicing
    ├─ Compression (40-60% smaller)
    └─ Metadata support

Access pattern:
  sample = hf['sample_000']['expert']['face'][:]  # Load ONLY this
  slice = hf['sample_000']['expert']['face'][100:200]  # Slice directly
```

## Performance Comparison

### Loading Time
```
Operation: Load sample 50

Pickle:
  ┌─────────────────────────────┐
  │ Load entire file: 2-5 sec   │ ████████████████
  ├─────────────────────────────┤
  │ Access sample 50: <0.01 sec │
  └─────────────────────────────┘
  Total: 2-5 seconds

HDF5:
  ┌─────────────────────────────┐
  │ Direct access: 0.05 sec     │ █
  └─────────────────────────────┘
  Total: 0.05 seconds

  → HDF5 is 50-100× faster! ✓
```

### Memory Usage
```
Dataset: 1000 samples, ~500MB total

Pickle:
  ┌────────────────────────────────────┐
  │ RAM: 500 MB (entire dataset)       │
  └────────────────────────────────────┘

HDF5:
  ┌──────────────────┐
  │ RAM: ~5 MB       │ (current sample only)
  └──────────────────┘

  → HDF5 uses 100× less memory! ✓
```

## Key Decision Points

### Q1: Should I use min-max scaling on ViT features?
```
NO! ❌

Before min-max:
  [0.5, -0.3, 0.8, -0.1, ...]  mean≈0, semantic meaning ✓

After min-max:
  [0.7, 0.0, 1.0, 0.2, ...]    mean≈0.5, semantics distorted ✗
```

### Q2: Should I use standardization on ViT features?
```
Usually NO, unless:
  ✓ Combining with other features (but use two-stream instead)
  ✓ Specific model requirements
  ✓ Domain adaptation (rare)

Default: Use as-is ✓
```

### Q3: Can I concatenate OpenFace and ViT directly?
```
NO (not recommended) ❌

OpenFace: [0.2, 0.5, 0.1, ...]     range [0, 1]
ViT:      [0.5, -1.2, 3.4, ...]    range [-4, 4]

→ ViT dominates gradients!

Solution: Use two-stream architecture ✓
  - Separate branches
  - Each branch handles its distribution
  - Fuse after initial processing
```

## Quick Reference

| Feature Type | Normalization | Code |
|--------------|---------------|------|
| OpenFace | Min-max [0, 1] | `data_hdf5.py` |
| ViT | None (as-is) | `data_hdf5_flexible.py` with `vit_normalization='none'` |
| Mixed | Per-feature | `data_hdf5_flexible.py` with `feature_type='mixed'` |

## Next Steps

1. **Test installation**
   ```bash
   python test_installation.py
   ```

2. **Choose your workflow** (see decision tree above)

3. **Extract features**
   - OpenFace: Already in CSV
   - ViT: Run `extract_vit_features.py`

4. **Create HDF5**
   - Run appropriate script

5. **Start training**
   - Use `pytorch_example.py` as template

## Summary

```
┌─────────────────────────────────────────────────────┐
│  Golden Rules                                       │
├─────────────────────────────────────────────────────┤
│  1. OpenFace → Min-max [0, 1]             ✓        │
│  2. ViT latents → Use as-is               ✓        │
│  3. Mixed → Two-stream architecture       ✓        │
│  4. HDF5 → Better than pickle             ✓        │
│  5. When in doubt → Keep original         ✓        │
└─────────────────────────────────────────────────────┘
```

For more details, see:
- [README.md](README.md) - Complete overview
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - Deep dive
