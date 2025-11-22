# Skeleton Stream Conversion Scripts

## Overview

Two scripts are available to convert skeleton streams to numpy arrays:

1. **[convert_all_streams.py](convert_all_streams.py)** - Convert to full 350-dim format
2. **[convert_streams_positions_only.py](convert_streams_positions_only.py)** - Extract positions only (75 dims)

## Which One Should I Use?

### Option 1: Full Data (350 dims) - Recommended for First Time
```bash
python convert_all_streams.py
```

**Output:**
- Location: `../data/processed_data/skeleton/`
- Format: `{sample}_{person}_skel.npy`
- Shape: `(T, 350)`
- Contains: All features (positions, orientations, metadata)

**Use when:**
- You're unsure what features you'll need
- You want maximum flexibility
- You can extract positions later if needed

### Option 2: Positions Only (75 dims) - Recommended for Production
```bash
python convert_streams_positions_only.py
```

**Output:**
- Location: `../data/processed_data/skeleton_positions/`
- Format: `{sample}_{person}_positions.npy`
- Shape: `(T, 75)`
- Contains: Only x,y,z positions for 25 joints

**Use when:**
- You only need body pose/movement
- You want to reduce file size (78% smaller)
- You want faster processing

## Detailed Usage

### Convert All Streams (Full Data)

```bash
cd /home/s2020425/NoXiRe/data_preparation
python convert_all_streams.py
```

**What it does:**
1. Searches for all `*.skel.stream~` files in `../data/aria-noxi`
2. Converts each to numpy array (T, 350)
3. Saves to `../data/processed_data/skeleton/`
4. Shows progress and summary

**Output files:**
```
data/processed_data/skeleton/
├── 001_2016-03-17_Paris_expert_skel.npy
├── 001_2016-03-17_Paris_novice_skel.npy
├── 002_2016-03-17_Paris_expert_skel.npy
└── ... (162 files total)
```

**Data structure (350 dims):**
```
Per frame: 25 joints × 14 features = 350 values

Per joint (14 features):
  [0-2]:   Position (x, y, z) in meters
  [3-6]:   Orientation quaternion (x, y, z, w)
  [7-13]:  Tracking metadata (confidence, state, etc.)
```

### Convert Positions Only

```bash
cd /home/s2020425/NoXiRe/data_preparation
python convert_streams_positions_only.py
```

**What it does:**
1. Searches for all `*.skel.stream~` files
2. Converts to numpy (350 dims)
3. **Extracts positions only** (reshapes to 25 × 3, then flattens to 75)
4. Saves to `../data/processed_data/skeleton_positions/`

**Output files:**
```
data/processed_data/skeleton_positions/
├── 001_2016-03-17_Paris_expert_positions.npy
├── 001_2016-03-17_Paris_novice_positions.npy
└── ... (162 files total)
```

**Data structure (75 dims):**
```
Per frame: 25 joints × 3 coordinates = 75 values

Layout (flattened):
  [0-2]:   SpineBase (x, y, z)
  [3-5]:   SpineMid (x, y, z)
  [6-8]:   Neck (x, y, z)
  [9-11]:  Head (x, y, z)
  [12-14]: ShoulderLeft (x, y, z)
  ...
  [72-74]: ThumbRight (x, y, z)
```

## Extract Positions from Full Data (Later)

If you already converted to 350 dims and want to extract positions:

```python
import numpy as np

# Load full data
data = np.load('path/to/sample_expert_skel.npy')  # (T, 350)

# Extract positions
T = data.shape[0]
reshaped = data.reshape(T, 25, 14)
positions = reshaped[:, :, :3]  # (T, 25, 3)
positions_flat = positions.reshape(T, 75)  # (T, 75)

# Save
np.save('path/to/sample_expert_positions.npy', positions_flat)
```

## File Size Comparison

### Full Data (350 dims)
```
Example: 1000 frames
Size: 1000 × 350 × 4 bytes = 1.4 MB per file
Total (162 files): ~227 MB
```

### Positions Only (75 dims)
```
Example: 1000 frames
Size: 1000 × 75 × 4 bytes = 0.3 MB per file
Total (162 files): ~49 MB (78% smaller!)
```

## Verification

After conversion, verify the output:

```python
import numpy as np

# Load a file
data = np.load('../data/processed_data/skeleton/001_2016-03-17_Paris_expert_skel.npy')

print(f"Shape: {data.shape}")  # Should be (T, 350)
print(f"Frames: {data.shape[0]}")
print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")

# Check positions
positions = data.reshape(-1, 25, 14)[:, :, :3]
print(f"Position X range: [{positions[:,:,0].min():.3f}, {positions[:,:,0].max():.3f}]")
print(f"Position Y range: [{positions[:,:,1].min():.3f}, {positions[:,:,1].max():.3f}]")
print(f"Position Z range: [{positions[:,:,2].min():.3f}, {positions[:,:,2].max():.3f}]")
```

Expected output:
```
Shape: (T, 350)
Frames: <varies by sample>
Data range: <varies>
Position X range: [-2, 2] meters (left-right)
Position Y range: [-1, 2] meters (bottom-top)
Position Z range: [0.5, 4.5] meters (depth)
```

## Integration with Pipeline

### PyTorch Dataset Example

```python
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, data_dir, use_positions_only=True):
        """
        Args:
            data_dir: Path to skeleton or skeleton_positions folder
            use_positions_only: True for 75 dims, False for 350 dims
        """
        self.data_dir = data_dir
        self.use_positions_only = use_positions_only
        self.files = sorted(glob.glob(f'{data_dir}/*.npy'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # (T, 75) or (T, 350)
        return torch.from_numpy(data).float()

# Usage
dataset = SkeletonDataset('../data/processed_data/skeleton_positions')
```

### With HDF5 Pipeline

See [hdf5_pipeline/](hdf5_pipeline/) for converting numpy arrays to HDF5 format.

## Troubleshooting

### "No skeleton stream files found"
- Check that `../data/aria-noxi` exists
- Verify files are named `expert.skel.stream~` or `novice.skel.stream~`

### "Input directory not found"
- Make sure you're running from `data_preparation/` folder
- Or adjust the `root_dir` path in the script

### "ModuleNotFoundError: No module named 'numpy'"
- Install numpy: `pip install numpy`
- Or use the environment that has numpy installed

### Files too large
- Use positions-only conversion (75 dims instead of 350)
- Or compress with HDF5 (see hdf5_pipeline/)

## Summary

**Quick Decision Guide:**

| Need | Script | Output Dims | File Size |
|------|--------|-------------|-----------|
| Everything (first time) | convert_all_streams.py | 350 | ~227 MB |
| Pose/movement only | convert_streams_positions_only.py | 75 | ~49 MB |
| Extract later | Load 350, extract positions | 75 | Manual |

**Recommended workflow:**
1. Test with [skeleton_testing/test_stream_conversion.py](skeleton_testing/test_stream_conversion.py)
2. Visualize with [skeleton_testing/visualize_skeleton.ipynb](skeleton_testing/visualize_skeleton.ipynb)
3. Convert all with **positions only** (most use cases)
4. Or convert all with **full data** (if unsure)

---

**Related Documentation:**
- [STREAM_CONVERSION_README.md](STREAM_CONVERSION_README.md) - Main conversion guide
- [skeleton_testing/README.md](skeleton_testing/README.md) - Testing guide
- [format_conversion.py](format_conversion.py) - Core converter class
