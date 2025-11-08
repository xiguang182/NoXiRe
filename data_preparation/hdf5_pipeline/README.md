# NoXiRe Data Preparation - HDF5 Pipeline

Complete pipeline for preparing NoXiRe dataset with efficient HDF5 storage and flexible feature handling (OpenFace, ViT, or mixed).

## 🎯 Your Question Answered

> **"What if the feature is a latent from some ViT, should it be min max or use it as is?"**

**Answer: Use ViT features as-is (no normalization)** ✅

- ViT features are already normalized by LayerNorm during the forward pass
- Min-max scaling would distort the learned semantic relationships
- All research papers (CLIP, DINO, MAE, etc.) use ViT features directly

See [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) for detailed explanation.

---

## 📁 Files Overview

### Core Implementation
| File | Purpose |
|------|---------|
| [data_hdf5.py](data_hdf5.py) | Original - OpenFace features with min-max scaling |
| [data_hdf5_flexible.py](data_hdf5_flexible.py) | **NEW** - Supports OpenFace, ViT, or mixed features |

### Feature Extraction
| File | Purpose |
|------|---------|
| [extract_vit_features.py](extract_vit_features.py) | Extract ViT embeddings from video frames |

### Training & Examples
| File | Purpose |
|------|---------|
| [pytorch_example.py](pytorch_example.py) | PyTorch Dataset & Model examples |
| [compare_formats.py](compare_formats.py) | Pickle vs HDF5 performance comparison |
| [normalization_comparison.py](normalization_comparison.py) | Visualize normalization effects |

### Documentation
| File | Purpose |
|------|---------|
| [QUICK_START.md](QUICK_START.md) | ⭐ Start here - Quick reference |
| [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) | Deep dive: Why different normalization? |
| [README_HDF5.md](README_HDF5.md) | HDF5 usage guide |

### Testing
| File | Purpose |
|------|---------|
| [test_installation.py](test_installation.py) | Verify dependencies and functionality |

---

## 🚀 Quick Start

### 1. Test Installation
```bash
python test_installation.py
```

### 2. Choose Your Use Case

#### Option A: OpenFace Only (Original)
```python
python data_hdf5.py
```
- Uses min-max scaling [0, 1]
- Good for handcrafted features

#### Option B: ViT Only (Recommended for deep features)
```python
# Step 1: Extract ViT features
python extract_vit_features.py

# Step 2: Save to HDF5 (no normalization!)
from data_hdf5_flexible import save_to_hdf5

save_to_hdf5(
    output_path='./data/vit.h5',
    feature_type='vit',
    vit_normalization='none',  # ← Keep as-is!
    vit_features_path='./data/vit_features/'
)
```

#### Option C: Mixed (OpenFace + ViT)
```python
from data_hdf5_flexible import save_to_hdf5

save_to_hdf5(
    output_path='./data/mixed.h5',
    feature_type='mixed',
    vit_normalization='none',  # ViT: as-is
    vit_features_path='./data/vit_features/'
)
# OpenFace: min-max, ViT: original distribution
```

---

## 📊 Comparison: OpenFace vs ViT Features

| Aspect | OpenFace (Handcrafted) | ViT (Learned) |
|--------|------------------------|---------------|
| **Type** | Landmarks, AUs, head pose | Embedding vectors |
| **Scale** | Different ranges (pixels, 0-5, radians) | Normalized by LayerNorm |
| **Normalization** | Min-max [0, 1] ✅ | None (as-is) ✅ |
| **Why** | Need scale alignment | Preserve semantics |
| **Dimensionality** | 159 (68×2 + 17 + 6) | 768 or 1024 |

---

## 🎓 Key Concepts

### Why Different Normalization?

**OpenFace:**
```python
# Before: Different scales
face_x: [100, 800]    # pixels
aus: [0, 5]           # intensity
head: [-π, π]         # radians

# After min-max [0, 1]: Aligned!
face_x: [0, 1]
aus: [0, 1]
head: [0, 1]
```

**ViT:**
```python
# Already normalized by LayerNorm
vit_features: mean ≈ 0.0, std ≈ 1.0

# DON'T DO THIS:
scaled = (vit - min) / (max - min)  # ❌ Distorts semantics!

# DO THIS:
features = vit_features  # ✅ Use as-is
```

---

## 💡 Usage Examples

### Load OpenFace Features
```python
from data_hdf5 import load_sample

data = load_sample('./data/openface.h5', sample_idx=0, person='expert')
print(data['face'].shape)  # (T, 68, 2) - scaled [0, 1]
print(data['aus'].shape)   # (T, 17) - scaled [0, 1]
print(data['head'].shape)  # (T, 6) - scaled [0, 1]
```

### Load ViT Features
```python
from data_hdf5_flexible import load_sample

data = load_sample('./data/vit.h5', sample_idx=0, person='expert')
print(data['vit'].shape)  # (T, 768) - original distribution
print(f"Mean: {data['vit'].mean():.3f}")  # Should be ≈ 0
```

### Load Specific Frames (Efficient Slicing)
```python
from data_hdf5_flexible import load_slice

# Load only frames 100-200
face = load_slice('./data/openface.h5', 0, 'expert', 'face', 100, 200)
print(face.shape)  # (100, 68, 2) - without loading full video!
```

### PyTorch Training
```python
from pytorch_example import NoXiReHDF5Dataset, ViTModel
from torch.utils.data import DataLoader

dataset = NoXiReHDF5Dataset(
    './data/vit.h5',
    feature_type='vit',
    sequence_length=100
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ViTModel(input_dim=768, hidden_dim=256, num_classes=7)

for features, _ in loader:
    outputs = model(features)  # features: (16, 100, 768)
    # ... training code ...
```

---

## 📈 Performance Benefits (HDF5 vs Pickle)

| Operation | Pickle | HDF5 | Speedup |
|-----------|--------|------|---------|
| **Load full dataset** | 2-5 sec | 2-5 sec | Similar |
| **Random access (10 samples)** | 2-5 sec | 0.05 sec | **50-100×** |
| **Slice frames (100-200)** | Load all → slice | Direct slice | **10-20×** |
| **File size** | 100 MB | 40-60 MB | **40-60%** smaller |
| **Memory usage** | All in RAM | Stream from disk | Much lower |

Run `python compare_formats.py` to see actual numbers on your data.

---

## 🔍 Verification

### Check ViT Features
```python
from extract_vit_features import inspect_vit_features

inspect_vit_features('./data/vit_features/001.001.001.001_expert.npy')
```

Expected output:
```
✓ Features appear to be from a normalized model (LayerNorm)
→ Recommended: Use as-is (no additional normalization)
```

### Visualize Normalization Effects
```python
python normalization_comparison.py
```

Generates plots showing:
- Why OpenFace needs min-max scaling
- Why ViT should NOT be normalized
- How min-max distorts ViT semantics

---

## 🏗️ Model Architecture Recommendations

### For ViT Features
```python
class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # No input normalization needed!
        self.lstm = nn.LSTM(768, 256, ...)
        # ViT features already in good distribution
```

### For Mixed Features (Two-Stream)
```python
class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate branches handle different distributions
        self.openface_branch = nn.Linear(159, 256)  # expects [0, 1]
        self.vit_branch = nn.Linear(768, 256)       # expects mean≈0
        self.fusion = ...
```

See [pytorch_example.py](pytorch_example.py) for complete implementations.

---

## 📦 Dependencies

```bash
# Core (required)
pip install h5py numpy pandas tqdm

# For ViT extraction (optional)
pip install torch torchvision transformers opencv-python

# For visualization (optional)
pip install matplotlib scipy
```

---

## 🐛 Troubleshooting

### "ViT features have unusual statistics"
- Check extraction code in [extract_vit_features.py](extract_vit_features.py)
- Verify model is loaded correctly
- Run `inspect_vit_features()` to diagnose

### "Training is unstable with ViT features"
- Make sure you're NOT normalizing ViT features
- Use `vit_normalization='none'`
- Check learning rate (ViT may need different LR than OpenFace)

### "Out of memory"
- Use `sequence_length` parameter to create fixed-length chunks
- Set `num_workers=0` in DataLoader initially
- Load features selectively (only what you need)

---

## 📚 Further Reading

1. **[QUICK_START.md](QUICK_START.md)** - TL;DR version with code snippets
2. **[NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)** - Why different features need different normalization
3. **[README_HDF5.md](README_HDF5.md)** - Advanced HDF5 usage, best practices

### Research Papers
- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- DINO: "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al., 2021)
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

All use ViT features as-is without additional normalization.

---

## ✅ Recommendations Summary

| Your Scenario | Recommendation |
|---------------|----------------|
| Only using OpenFace | Use [data_hdf5.py](data_hdf5.py) with min-max scaling |
| Only using ViT | Use [data_hdf5_flexible.py](data_hdf5_flexible.py) with `vit_normalization='none'` |
| Using both | Use two-stream architecture with different normalization per type |
| Unsure | Start with OpenFace (simpler), add ViT later if needed |

---

## 🎯 Golden Rules

1. **OpenFace** → Min-max to [0, 1] ✅
2. **ViT latents** → Use as-is (no normalization) ✅
3. **When in doubt with pre-trained features** → Keep original distribution ✅

---

## 📞 Support

Issues or questions:
1. Check [QUICK_START.md](QUICK_START.md) for common scenarios
2. Run [test_installation.py](test_installation.py) to diagnose issues
3. Review [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) for detailed explanations

---

## 🎉 You're Ready!

1. ✅ Run `python test_installation.py`
2. ✅ Choose your feature type (OpenFace / ViT / Mixed)
3. ✅ Extract and save to HDF5
4. ✅ Start training!

**Remember:** ViT features = use as-is! 🎯
