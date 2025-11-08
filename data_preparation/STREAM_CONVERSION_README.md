# Skeleton Stream Conversion

## Overview

The `format_conversion.py` script converts binary skeleton stream files (`.skel.stream~`) from the NoXiRe dataset into numpy arrays for efficient processing.

## What It Does

### Input
- **Binary stream files**: `expert.skel.stream~` and `novice.skel.stream~`
- **Location**: `/home/s2020425/NoXiRe/data/aria-noxi/[sample_name]/`
- **Format**: Binary floats (4 bytes each)
- **Dimensions**: 350 features per frame

### Output
- **Numpy arrays**: `.npy` files
- **Location**: `/home/s2020425/NoXiRe/data/processed_skel/`
- **Shape**: `(T, 350)` where T = number of frames
- **Naming**: `{sample}_{person}_skel.npy`
  - Example: `001_2016-03-17_Paris_expert_skel.npy`

## Data Structure

```
NoXiRe/
├── data/
│   ├── aria-noxi/                    # Input: Binary streams
│   │   ├── 001_2016-03-17_Paris/
│   │   │   ├── expert.skel.stream~   # 350-dim skeleton
│   │   │   └── novice.skel.stream~
│   │   ├── 002_2016-03-17_Paris/
│   │   └── ...
│   │
│   └── processed_skel/               # Output: Numpy arrays
│       ├── 001_2016-03-17_Paris_expert_skel.npy
│       ├── 001_2016-03-17_Paris_novice_skel.npy
│       └── ...
│
└── data_preparation/
    └── format_conversion.py          # This script
```

## Usage

### Basic Usage (Convert All Files)

```python
from format_conversion import StreamConverter

# Initialize converter
converter = StreamConverter(
    root_dir='../data/aria-noxi',
    save_dir='../data/processed_skel',
    feature_dim=350  # Skeleton dimension
)

# Convert all skel.stream~ files
results = converter.convert_all_streams(pattern='*skel.stream~', verbose=True)

# Verify conversion
converter.verify_conversion(sample_idx=0)
```

### Run from Command Line

```bash
cd /home/s2020425/NoXiRe/data_preparation
python format_conversion.py
```

### Advanced Usage

```python
from format_conversion import StreamConverter

converter = StreamConverter()

# Convert single file
stream_path = '../data/aria-noxi/001_2016-03-17_Paris/expert.skel.stream~'
data, save_path = converter.convert_stream_to_numpy(stream_path, save=True)
print(f"Shape: {data.shape}")  # (T, 350)

# Search for specific files
all_streams = converter.search_stream_files('*skel.stream~')
print(f"Found {len(all_streams)} files")

# Get conversion summary
summary = converter.get_conversion_summary()
print(summary)
```

## Output Format

### Numpy Array Properties
```python
import numpy as np

# Load converted file
data = np.load('../data/processed_skel/001_2016-03-17_Paris_expert_skel.npy')

# Properties
print(data.shape)    # (T, 350) - T frames, 350 features
print(data.dtype)    # float32
print(data.nbytes)   # Memory size in bytes

# Access data
first_frame = data[0]           # (350,) - first frame
first_10_frames = data[:10]     # (10, 350) - first 10 frames
feature_5 = data[:, 5]          # (T,) - feature 5 across all frames
```

## Key Features

### 1. **Robust Binary Reading**
- Reads 4-byte floats using `struct.unpack('f', ...)`
- Handles incomplete samples (corrupted files)
- Progress bar with tqdm

### 2. **Flexible Configuration**
- Configurable feature dimensions
- Custom input/output directories
- Pattern-based file search

### 3. **Automatic Naming**
- Generates descriptive output names
- Example: `expert.skel.stream~` → `001_2016-03-17_Paris_expert_skel.npy`

### 4. **Verification Tools**
- `verify_conversion()` - Check loaded arrays
- `get_conversion_summary()` - Dataset statistics

## Expected Dataset

Based on the notebook, you should have:
- **~46 samples** (each with expert and novice)
- **~92 skel.stream~ files** total
- **350 features** per frame (skeleton joints/keypoints)

## Comparison with Original Notebook

### Original (format_conversion.ipynb)
```python
# Converted multiple stream types:
# - au.stream~ (17 dim)
# - face.stream~ (4041 dim)
# - head.stream~ (3 dim)
# - skel.stream~ (350 dim)

# Then merged with engagement/smile/headshake data
```

### New Script (format_conversion.py)
```python
# Focuses ONLY on skel.stream~ (350 dim)
# Clean, reusable class-based design
# Direct binary → numpy conversion
# No CSV intermediate step
```

## Next Steps

After running this conversion, you can:

1. **Integrate with HDF5 pipeline** (like we did for OpenFace/ViT)
2. **Use directly in PyTorch DataLoader**
3. **Apply normalization** (discuss: standardize vs min-max)
4. **Combine with other features** (OpenFace, ViT, etc.)

## Example Workflow

```python
from format_conversion import StreamConverter

# Step 1: Convert binary streams to numpy
converter = StreamConverter()
results = converter.convert_all_streams()

# Step 2: Load and inspect
import numpy as np
skel_data = np.load('../data/processed_skel/001_2016-03-17_Paris_expert_skel.npy')
print(f"Shape: {skel_data.shape}")
print(f"Sample frame: {skel_data[0, :5]}")  # First 5 features of first frame

# Step 3: Use in your pipeline
# (We can discuss integration with HDF5 or PyTorch)
```

## Performance

- **Conversion speed**: ~15 seconds per file (depending on size)
- **Output size**: ~5-15 MB per file (depends on video length)
- **Memory efficient**: Processes one file at a time

## Troubleshooting

### Issue: "FileNotFoundError: Root directory not found"
- Check that `../data/aria-noxi` exists
- Or specify custom path: `StreamConverter(root_dir='/path/to/aria-noxi')`

### Issue: "Incomplete feature sample"
- Some stream files may be corrupted
- Script will skip and continue with next file
- Check the warning message for which file

### Issue: "No files found"
- Verify files exist: `ls ../data/aria-noxi/*/expert.skel.stream~`
- Check pattern matches your files

## Questions for Discussion

Now that we have clean skeleton stream conversion, we should discuss:

1. **Normalization**: Should skeleton data be:
   - Standardized (z-score)?
   - Min-max scaled?
   - Used as-is?

2. **Integration**: Do you want to:
   - Keep separate from HDF5 pipeline?
   - Integrate into `hdf5_pipeline/`?
   - Create a unified multi-modal dataset?

3. **Features**: Are these 350 dimensions:
   - 3D joint positions (x, y, z)?
   - Joint positions + confidence scores?
   - Normalized or raw coordinates?

4. **Next steps**: Should we:
   - Create HDF5 version for skeleton data?
   - Combine skeleton + OpenFace + ViT?
   - Create PyTorch Dataset for skeleton data?
