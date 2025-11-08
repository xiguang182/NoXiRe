# Quick Start Guide

## Summary: OpenFace vs ViT Features

| Aspect | OpenFace (Handcrafted) | ViT (Learned Embeddings) |
|--------|------------------------|--------------------------|
| **Normalization** | Min-max [0, 1] ✓ | None (use as-is) ✓ |
| **Why** | Different scales (pixels, angles) | Already normalized by LayerNorm |
| **File** | [data_hdf5.py](data_hdf5.py) | [data_hdf5_flexible.py](data_hdf5_flexible.py) |

## Your Question Answered

> What if the feature is a latent from some ViT, should it be min max or use it as is?

**Answer: Use ViT features as-is (no normalization)** ✓

### Why?

1. **ViT already applies LayerNorm** during forward pass
2. **Semantic relationships** are encoded in the embedding space
3. **Min-max would distort** the learned representation
4. **All research papers** use ViT features directly (CLIP, DINO, etc.)

## File Overview

### Core Implementation
- **[data_hdf5.py](data_hdf5.py)** - Original OpenFace features (min-max scaling)
- **[data_hdf5_flexible.py](data_hdf5_flexible.py)** - NEW: Supports OpenFace, ViT, or mixed features

### Feature Extraction
- **[extract_vit_features.py](extract_vit_features.py)** - Extract ViT embeddings from videos

### Training Examples
- **[pytorch_example.py](pytorch_example.py)** - PyTorch training with different feature types

### Documentation
- **[NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)** - Deep dive on normalization choices
- **[README_HDF5.md](README_HDF5.md)** - HDF5 usage guide
- **[compare_formats.py](compare_formats.py)** - Pickle vs HDF5 comparison

## Usage Examples

### 1. OpenFace Only (Your Current Setup)

```python
from data_hdf5 import save_to_hdf5, load_sample

# Save with min-max scaling
save_to_hdf5('./data/openface.h5')

# Load
data = load_sample('./data/openface.h5', sample_idx=0, person='expert')
# data['face']: (T, 68, 2) - scaled [0, 1]
# data['aus']: (T, 17) - scaled [0, 1]
# data['head']: (T, 6) - scaled [0, 1]
```

### 2. ViT Only (Recommended: No Normalization)

```python
from data_hdf5_flexible import save_to_hdf5, load_sample

# Step 1: Extract ViT features
from extract_vit_features import extract_all_samples
extract_all_samples(
    video_folder='./data/videos/',
    sample_list_csv='./data/sample_list.csv',
    output_folder='./data/vit_features/',
    model_name='google/vit-base-patch16-224'
)

# Step 2: Save to HDF5 (no normalization!)
save_to_hdf5(
    output_path='./data/vit.h5',
    feature_type='vit',
    vit_normalization='none',  # ← Keep as-is!
    vit_features_path='./data/vit_features/'
)

# Step 3: Load
data = load_sample('./data/vit.h5', sample_idx=0, person='expert')
# data['vit']: (T, 768) - original ViT distribution (mean≈0, std≈1)
```

### 3. Mixed Features (OpenFace + ViT)

```python
# Save both feature types
save_to_hdf5(
    output_path='./data/mixed.h5',
    feature_type='mixed',
    vit_normalization='none',  # Keep ViT as-is
    vit_features_path='./data/vit_features/'
)

# Load
data = load_sample('./data/mixed.h5', sample_idx=0, person='expert')
# data['face']: (T, 68, 2) - min-max scaled
# data['aus']: (T, 17) - min-max scaled
# data['head']: (T, 6) - min-max scaled
# data['vit']: (T, 768) - original distribution
```

## PyTorch Integration

### OpenFace Model
```python
from pytorch_example import NoXiReHDF5Dataset, OpenFaceModel

dataset = NoXiReHDF5Dataset(
    './data/openface.h5',
    feature_type='openface',
    sequence_length=100
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = OpenFaceModel(input_dim=159, hidden_dim=256, num_classes=7)

# Train...
for features, _ in dataloader:
    # features: (16, 100, 159) - min-max scaled [0, 1]
    outputs = model(features)
```

### ViT Model
```python
from pytorch_example import NoXiReHDF5Dataset, ViTModel

dataset = NoXiReHDF5Dataset(
    './data/vit.h5',
    feature_type='vit',
    sequence_length=100
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ViTModel(input_dim=768, hidden_dim=256, num_classes=7)

# Train...
for features, _ in dataloader:
    # features: (16, 100, 768) - original ViT distribution
    outputs = model(features)
```

### Mixed Model (Two-Stream)
```python
from pytorch_example import NoXiReHDF5Dataset, MixedModel

dataset = NoXiReHDF5Dataset(
    './data/mixed.h5',
    feature_type='mixed',
    sequence_length=100
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = MixedModel(openface_dim=159, vit_dim=768, hidden_dim=256, num_classes=7)

# Train...
for features, _ in dataloader:
    # features['openface']: (16, 100, 159) - min-max scaled
    # features['vit']: (16, 100, 768) - original distribution
    outputs = model(features['openface'], features['vit'])
```

## Verification

### Check ViT Features Are Properly Extracted

```python
from extract_vit_features import inspect_vit_features

inspect_vit_features('./data/vit_features/001.001.001.001_expert.npy')
```

Expected output:
```
Shape: (T, 768)
Statistics:
  Mean: ≈ 0.0      ← Should be close to 0
  Std:  ≈ 1.0      ← Should be 0.5-2.0
  Min:  ≈ -3 to -5
  Max:  ≈ 3 to 5

Normalization check:
  ✓ Features appear to be from a normalized model (LayerNorm)
  → Recommended: Use as-is (no additional normalization)
```

## Key Takeaways

1. **OpenFace features** → Min-max scale to [0, 1]
   - Different physical units need alignment
   - Code: [data_hdf5.py](data_hdf5.py)

2. **ViT features** → Use as-is (no normalization)
   - Already normalized by LayerNorm
   - Preserves semantic relationships
   - Code: [data_hdf5_flexible.py](data_hdf5_flexible.py) with `vit_normalization='none'`

3. **Mixed features** → Different normalization per type
   - Use two-stream architecture
   - Each branch handles its input distribution
   - Code: [pytorch_example.py](pytorch_example.py) `MixedModel`

## Dependencies

```bash
# Core
pip install h5py numpy pandas tqdm

# For ViT extraction
pip install torch torchvision transformers opencv-python

# For training
pip install pytorch-lightning  # optional
```

## Next Steps

1. **For OpenFace only**: Use [data_hdf5.py](data_hdf5.py) (already done ✓)

2. **To add ViT features**:
   - Run [extract_vit_features.py](extract_vit_features.py) to extract embeddings
   - Use [data_hdf5_flexible.py](data_hdf5_flexible.py) with `vit_normalization='none'`
   - Train with [pytorch_example.py](pytorch_example.py)

3. **Read the guides**:
   - [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) - Why different normalization?
   - [README_HDF5.md](README_HDF5.md) - How to use HDF5 efficiently?

## Questions?

- **"Should I standardize ViT features?"** → No, unless you have a specific reason (see [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md))
- **"Can I use min-max on ViT?"** → Not recommended, distorts semantic meaning
- **"How to combine OpenFace + ViT?"** → Use two-stream model (see [pytorch_example.py](pytorch_example.py))
- **"Which ViT model to use?"** → `google/vit-base-patch16-224` or `facebook/dino-vitb16` (see [extract_vit_features.py](extract_vit_features.py))

## Recommendation for Your Project

Based on your question, I recommend:

1. **Keep your current OpenFace processing** with min-max scaling ✓
2. **If adding ViT features**: Extract them and **use as-is** (no normalization)
3. **Use HDF5** for efficient storage and slicing
4. **Use two-stream architecture** if combining both feature types

The flexible implementation in [data_hdf5_flexible.py](data_hdf5_flexible.py) supports all these scenarios!
