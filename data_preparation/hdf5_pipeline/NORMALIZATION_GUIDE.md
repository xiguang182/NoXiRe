# Feature Normalization Guide

## TL;DR

- **OpenFace (handcrafted features)**: Use min-max scaling ✓
- **ViT latents (learned embeddings)**: Use as-is (no normalization) ✓
- **Mixed features**: Different normalization per feature type ✓

## Why Different Normalization?

### OpenFace Features (Handcrafted)

**What they are:**
- Facial landmarks: (x, y) pixel coordinates
- Action Units (AUs): Intensity values 0-5
- Head pose: Position (x, y, z) and rotation (pitch, yaw, roll)

**Why min-max scaling:**
```
Original:
  face_x: [0, 1920]      # pixels
  face_y: [0, 1080]      # pixels
  aus: [0, 5]            # intensity
  head_pos: [-100, 100]  # arbitrary units

After min-max [0, 1]:
  face_x: [0, 1]
  face_y: [0, 1]
  aus: [0, 1]
  head_pos: [0, 1]
```

**Benefits:**
- Brings all features to same scale [0, 1]
- Makes different feature types comparable
- Improves neural network training stability
- Prevents features with large ranges from dominating

### ViT Latent Features (Learned Embeddings)

**What they are:**
- Output from Vision Transformer's encoder
- Pre-trained on massive datasets (ImageNet, etc.)
- Already in a meaningful, learned embedding space
- Typically shape: (num_frames, embedding_dim) where embedding_dim = 768, 1024, etc.

**Why use as-is (no normalization):**

1. **Already normalized during training**
   - ViT uses LayerNorm internally
   - Embeddings are in a stable distribution
   - Mean ≈ 0, reasonable variance

2. **Semantic meaning preserved**
   - Distance in embedding space = semantic similarity
   - Min-max scaling distorts these relationships
   - Example: if all values are positive after LayerNorm, min-max forces minimum to 0

3. **Transfer learning works best**
   - Fine-tuning expects same distribution as pre-training
   - Changing normalization breaks learned representations

4. **Empirical evidence**
   - Research shows ViT features work best as-is
   - Used directly in papers (CLIP, DINO, MAE, etc.)

**Example:**
```python
# ViT output (already normalized internally)
vit_features = model.forward(images)  # (T, 768)
# mean ≈ 0.0, std ≈ 1.0 (approximately)

# DON'T DO THIS:
min_val = vit_features.min()  # e.g., -3.2
max_val = vit_features.max()  # e.g., 4.1
scaled = (vit_features - min_val) / (max_val - min_val)
# Now [0, 1] but semantic meaning is lost!

# DO THIS:
# Just use vit_features directly ✓
```

## Comparison Table

| Feature Type | Normalization | Why |
|--------------|---------------|-----|
| **OpenFace landmarks** | Min-max [0,1] | Pixel coordinates, need scale alignment |
| **OpenFace AUs** | Divide by 5 → [0,1] | Original range [0,5], known bounds |
| **OpenFace head pose** | Min-max [0,1] | Arbitrary units, need scale alignment |
| **ViT embeddings** | **None** (as-is) | Already normalized, preserve semantics |
| **ResNet features** | None or standardize | Pre-normalized via BatchNorm |
| **CLIP features** | **None** (as-is) | Pre-normalized to unit sphere |

## When You Might Standardize ViT Features

Use Z-score standardization `(x - mean) / std` **only if**:

1. **Combining with other features**
   - If concatenating ViT with other features
   - To ensure similar variance across feature types
   - But consider: maybe use separate branches instead

2. **Domain adaptation**
   - If your data distribution is very different from pre-training
   - But this is rare and requires careful validation

3. **Specific model requirements**
   - Some downstream models expect zero-mean features
   - Check your model documentation

**Example:**
```python
# Only if you have a good reason:
vit_features = (vit_features - vit_features.mean(axis=0)) / vit_features.std(axis=0)
```

## When Min-Max is Harmful for ViT

**Example scenario:**
```python
# Frame 1: ViT embedding for "person smiling"
frame1 = np.array([0.5, -0.3, 0.8, -0.1, ...])  # 768 dims

# Frame 2: ViT embedding for "person frowning"
frame2 = np.array([0.3, -0.5, 0.6, -0.3, ...])

# Frame 3: ViT embedding for "person neutral"
frame3 = np.array([0.4, -0.4, 0.7, -0.2, ...])

# Video: (3, 768)
video = np.stack([frame1, frame2, frame3])

# Min-max per video (WRONG!)
video_minmax = (video - video.min(axis=0)) / (video.max(axis=0) - video.min(axis=0))

# Problem: Now min=0, max=1 for EACH dimension
# - Relative distances are distorted
# - Semantic similarity is broken
# - Different videos have different scaling
# - Can't compare across videos!
```

## Usage Examples

### OpenFace Only (Original)
```python
from data_hdf5_flexible import save_to_hdf5

# Min-max scaling for all OpenFace features
save_to_hdf5(
    output_path='./data/openface.h5',
    feature_type='openface'
)
```

### ViT Only (Recommended: No Normalization)
```python
# Keep ViT embeddings as-is
save_to_hdf5(
    output_path='./data/vit.h5',
    feature_type='vit',
    vit_normalization='none',  # Recommended!
    vit_features_path='./data/vit_features/'
)
```

### ViT with Standardization (If Needed)
```python
# Use only if you have a specific reason
save_to_hdf5(
    output_path='./data/vit_standardized.h5',
    feature_type='vit',
    vit_normalization='standardize',  # Z-score
    vit_features_path='./data/vit_features/'
)
```

### Mixed Features (Different Normalization)
```python
# OpenFace: min-max, ViT: as-is
save_to_hdf5(
    output_path='./data/mixed.h5',
    feature_type='mixed',
    vit_normalization='none',  # Keep ViT as-is
    vit_features_path='./data/vit_features/'
)

# Load separately in your model
from data_hdf5_flexible import load_sample

data = load_sample('./data/mixed.h5', 0, 'expert')
openface = data['face']  # (T, 68, 2) - min-max scaled
vit = data['vit']        # (T, 768) - original distribution

# Use in separate branches
openface_branch = OpenFaceEncoder(openface)
vit_branch = ViTEncoder(vit)  # Expects original ViT distribution
combined = Fusion(openface_branch, vit_branch)
```

## Model Architecture Considerations

### Two-Stream Architecture (Recommended)
```python
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate branches for different feature types
        self.openface_branch = nn.Sequential(
            nn.Linear(68*2 + 17 + 6, 256),  # face + aus + head
            nn.ReLU(),
            # ... expects normalized [0,1] features
        )

        self.vit_branch = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            # ... expects ViT distribution (mean≈0)
        )

        self.fusion = nn.Linear(512, num_classes)

    def forward(self, openface_features, vit_features):
        x1 = self.openface_branch(openface_features)
        x2 = self.vit_branch(vit_features)
        return self.fusion(torch.cat([x1, x2], dim=-1))
```

### Why Not Concatenate Directly?
```python
# Bad: Concatenating differently-scaled features
combined = torch.cat([
    openface_features,  # [0, 1] range
    vit_features        # [-3, 3] range (approx)
], dim=-1)

# ViT features will dominate gradients!
# Loss = f(openface * w1 + vit * w2)
# ∂Loss/∂w2 >> ∂Loss/∂w1 due to larger range
```

## Testing Your Choice

```python
import numpy as np

# Load your ViT features
vit_features = np.load('sample_expert.npy')  # (T, D)

print("ViT feature statistics:")
print(f"  Shape: {vit_features.shape}")
print(f"  Mean: {vit_features.mean():.3f}")
print(f"  Std: {vit_features.std():.3f}")
print(f"  Min: {vit_features.min():.3f}")
print(f"  Max: {vit_features.max():.3f}")

# Should look something like:
#   Mean: ≈ 0.0 (close to zero)
#   Std: ≈ 0.5-2.0 (reasonable)
#   Min: ≈ -3 to -5
#   Max: ≈ 3 to 5
# This indicates LayerNorm is already applied

# Check normalization impact
minmax = (vit_features - vit_features.min()) / (vit_features.max() - vit_features.min())
print(f"\nAfter min-max:")
print(f"  Mean: {minmax.mean():.3f}")  # Will be ~0.5
print(f"  Std: {minmax.std():.3f}")     # Will be much smaller

# Semantic check: cosine similarity
from scipy.spatial.distance import cosine
sim_before = 1 - cosine(vit_features[0], vit_features[1])
sim_after = 1 - cosine(minmax[0], minmax[1])
print(f"\nCosine similarity frame 0 vs frame 1:")
print(f"  Before normalization: {sim_before:.3f}")
print(f"  After min-max: {sim_after:.3f}")
print(f"  Change: {abs(sim_before - sim_after):.3f}")
# If this changes significantly, normalization is distorting semantics!
```

## References

1. **ViT Paper**: "An Image is Worth 16x16 Words" - Uses LayerNorm, no post-normalization
2. **CLIP**: Features are L2-normalized to unit sphere, use as-is
3. **DINO**: Self-supervised ViT, uses features directly
4. **Facial Expression Recognition**: Typically use ViT features without normalization

## Summary

| Your Use Case | Recommendation |
|---------------|----------------|
| Only OpenFace | Use min-max (original [data_hdf5.py](data_hdf5.py)) |
| Only ViT | Use as-is, `vit_normalization='none'` |
| Mixed (OpenFace + ViT) | Min-max for OpenFace, none for ViT |
| Concatenating features | Use separate branches or standardize both |
| Fine-tuning ViT | Definitely keep as-is |
| Using frozen ViT | Keep as-is |

**Golden Rule**: When in doubt with pre-trained features → use as-is! 🎯
