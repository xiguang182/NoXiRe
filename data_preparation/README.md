# Data Preparation - NoXiRe Project

This directory contains scripts for preparing and converting data for the NoXiRe (Rapport Recognition) project.

## Directory Structure

```
data_preparation/
├── format_conversion.py              # Skeleton stream converter (binary → numpy)
├── data_pkl.py                       # Original pickle data loading
├── label_pkl.py                      # Label loading
├── hdf5_pipeline/                    # HDF5 conversion pipeline
│   ├── data_hdf5.py                  # OpenFace → HDF5
│   ├── data_hdf5_flexible.py         # Flexible (OpenFace/ViT/Mixed)
│   ├── extract_vit_features.py       # ViT feature extraction
│   ├── pytorch_example.py            # Training examples
│   └── [documentation files]
└── skeleton_testing/                 # Testing & visualization
    ├── test_stream_conversion.py     # Test suite
    ├── visualize_skeleton.ipynb      # Colab visualization
    └── [documentation files]
```

## Main Components

### 1. Skeleton Stream Conversion
**Purpose:** Convert Kinect skeleton binary streams to numpy arrays

**Main script:** [format_conversion.py](format_conversion.py)

```python
from format_conversion import StreamConverter

converter = StreamConverter(
    root_dir='../data/aria-noxi',
    save_dir='../data/processed_skel'
)
results = converter.convert_all_streams()
```

**Data structure:**
- Input: 162 binary `.skel.stream~` files (350 dimensions)
- Output: Numpy arrays (T, 350) or (T, 25, 14)
- Structure: 25 Kinect v2 joints × 14 features per joint
- Features: [x, y, z, qx, qy, qz, qw, ...tracking metadata...]

**Testing:** See [skeleton_testing/](skeleton_testing/) folder

---

### 2. HDF5 Pipeline
**Purpose:** Convert pickle/numpy data to HDF5 format for efficient storage and access

**Main scripts:**
- [hdf5_pipeline/data_hdf5.py](hdf5_pipeline/data_hdf5.py) - OpenFace features
- [hdf5_pipeline/data_hdf5_flexible.py](hdf5_pipeline/data_hdf5_flexible.py) - Multi-modal

**Benefits:**
- 50-100× faster random access
- 40-60% smaller file size
- Direct frame slicing
- Metadata support

**Documentation:** See [hdf5_pipeline/README.md](hdf5_pipeline/README.md)

---

### 3. Legacy Scripts
- [data_pkl.py](data_pkl.py) - Original pickle data loading
- [label_pkl.py](label_pkl.py) - Label loading
- [format_conversion.ipynb](format_conversion.ipynb) - Original notebook (reference)

---

## Quick Start Workflows

### Skeleton Stream Conversion

**1. Test first (recommended):**
```bash
cd skeleton_testing
python test_stream_conversion.py
```

**2. Visualize to verify:**
- Upload `skeleton_testing/visualize_skeleton.ipynb` to Google Colab
- Upload `skeleton_testing/test_conversion_output/sample_expert_skel.npy`
- Run all cells

**3. Full conversion:**
```bash
python format_conversion.py
```

### HDF5 Conversion

**1. Install dependencies:**
```bash
cd hdf5_pipeline
python test_installation.py
```

**2. Convert data:**
```python
from hdf5_pipeline.data_hdf5_flexible import save_to_hdf5

save_to_hdf5(
    data_dict={'openface': openface_data, 'skeleton': skeleton_data},
    normalization={'openface': 'minmax', 'skeleton': 'none'},
    output_path='data.h5'
)
```

**Documentation:** See [hdf5_pipeline/QUICK_START.md](hdf5_pipeline/QUICK_START.md)

---

## Data Flow

```
Binary Streams (.skel.stream~)
    ↓
[format_conversion.py]
    ↓
Numpy Arrays (.npy)
    ↓
[Optional: Extract positions only]
    ↓
HDF5 Files (.h5) ← [hdf5_pipeline/]
    ↓
PyTorch Dataset
    ↓
Training
```

---

## Feature Extraction Options

### Skeleton Features

**Option A: Positions only (75 dims)**
```python
# Extract x, y, z positions for 25 joints
data = np.load('skel.npy')
positions = data.reshape(-1, 25, 14)[:, :, :3]  # (T, 25, 3)
positions_flat = positions.reshape(-1, 75)      # (T, 75)
```
**Use for:** Basic pose/movement analysis

**Option B: Positions + Orientations (175 dims)**
```python
# Extract positions + quaternions
pose_data = data.reshape(-1, 25, 14)[:, :, :7]  # (T, 25, 7)
pose_flat = pose_data.reshape(-1, 175)           # (T, 175)
```
**Use for:** Head/body facing direction, advanced gestures

**Option C: Full data (350 dims)**
```python
# Keep everything
full_data = data  # (T, 350)
```
**Use for:** Uncertain future needs, includes tracking metadata

---

## Documentation

### Skeleton Conversion
- [STREAM_CONVERSION_README.md](STREAM_CONVERSION_README.md) - Main guide
- [skeleton_testing/README.md](skeleton_testing/README.md) - Testing guide
- [skeleton_testing/COLAB_VISUALIZATION_GUIDE.md](skeleton_testing/COLAB_VISUALIZATION_GUIDE.md) - Visualization

### HDF5 Pipeline
- [hdf5_pipeline/README.md](hdf5_pipeline/README.md) - Overview
- [hdf5_pipeline/NORMALIZATION_GUIDE.md](hdf5_pipeline/NORMALIZATION_GUIDE.md) - Feature normalization
- [hdf5_pipeline/QUICK_START.md](hdf5_pipeline/QUICK_START.md) - Quick start

---

## Dependencies

**Core (required):**
- Python 3.8+
- numpy

**HDF5 pipeline:**
- h5py
- torch (for PyTorch integration)

**ViT features:**
- transformers
- PIL

**Visualization:**
- matplotlib
- plotly

**Optional:**
- tqdm (progress bars)
- jupyter (local notebook support)

---

## File Locations

```
/home/s2020425/NoXiRe/
├── data/
│   ├── aria-noxi/              # Input: Binary streams
│   ├── processed_skel/         # Output: Numpy arrays (skeleton)
│   └── [other processed data]
│
└── data_preparation/           # This directory
    ├── format_conversion.py
    ├── hdf5_pipeline/
    └── skeleton_testing/
```

---

## Common Tasks

### Convert all skeleton streams
```bash
python format_conversion.py
```

### Test one file
```bash
cd skeleton_testing
python test_stream_conversion.py
```

### Compare format performance
```bash
cd hdf5_pipeline
python compare_formats.py
```

### Extract ViT features
```bash
cd hdf5_pipeline
python extract_vit_features.py --video_dir /path/to/videos --output_dir /path/to/output
```

---

## Next Steps

1. ✓ Test skeleton conversion
2. ✓ Visualize one sample
3. Run full conversion (162 files)
4. Decide on feature extraction strategy
5. Integrate with HDF5 pipeline (optional)
6. Create PyTorch Dataset

---

## Support

For issues or questions, see:
- [STREAM_CONVERSION_README.md](STREAM_CONVERSION_README.md) - Skeleton conversion
- [hdf5_pipeline/README.md](hdf5_pipeline/README.md) - HDF5 pipeline
- [skeleton_testing/](skeleton_testing/) - Testing & visualization

---

**Status:**
- ✅ Skeleton conversion: Ready to use
- ✅ Testing suite: Complete
- ✅ Visualization: Google Colab ready
- ✅ HDF5 pipeline: Available for integration
