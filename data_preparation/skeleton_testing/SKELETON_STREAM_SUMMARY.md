# Skeleton Stream Conversion - Complete Summary

## Files Created

### Core Files
1. **[format_conversion.py](format_conversion.py)** - Main conversion script
   - `StreamConverter` class for binary → numpy conversion
   - Handles 350-dimensional skeleton data
   - Robust error handling

2. **[test_stream_conversion.py](test_stream_conversion.py)** - Testing suite
   - Test 1: Load all 162 stream files
   - Test 2: Convert one sample file
   - Test 3: Analyze Kinect structure

### Documentation
3. **[STREAM_CONVERSION_README.md](STREAM_CONVERSION_README.md)** - Main guide
4. **[TEST_README.md](TEST_README.md)** - Testing guide
5. **[SKELETON_STREAM_SUMMARY.md](SKELETON_STREAM_SUMMARY.md)** - This file

## Quick Start

### Run Tests First (Recommended)
```bash
cd /home/s2020425/NoXiRe/data_preparation
python test_stream_conversion.py
```

**This will:**
- ✓ Verify all 162 stream files are readable
- ✓ Convert one sample for inspection
- ✓ Analyze the 350-dimension structure
- ✓ Save results to `./test_conversion_output/`

### Full Conversion (After Testing)
```python
from format_conversion import StreamConverter

converter = StreamConverter()
results = converter.convert_all_streams()
```

**This will:**
- Convert all 162 files to numpy arrays
- Save to `../data/processed_skel/`
- Each file: `{sample}_{person}_skel.npy` with shape (T, 350)

## Data Structure

### Input
```
../data/aria-noxi/
├── 001_2016-03-17_Paris/
│   ├── expert.skel.stream~  (binary, 350 dims per frame)
│   └── novice.skel.stream~
├── 002_2016-03-17_Paris/
│   └── ...
└── [81 samples total]
```

### Output
```
../data/processed_skel/
├── 001_2016-03-17_Paris_expert_skel.npy  (T, 350)
├── 001_2016-03-17_Paris_novice_skel.npy
└── [162 files total]
```

## Expected Structure: Kinect v2

### Hypothesis: 25 joints × 14 features = 350 dimensions

**Per joint (14 features):**
```
[0-2]:   Position (x, y, z) in meters
[3-6]:   Orientation quaternion (x, y, z, w)?
[7-13]:  Other features (confidence, tracking state, etc.)?
```

**25 Kinect v2 Joints:**
```
 0: SpineBase         13: HipLeft
 1: SpineMid          14: KneeLeft
 2: Neck              15: AnkleLeft
 3: Head              16: FootLeft
 4: ShoulderLeft      17: HipRight
 5: ElbowLeft         18: KneeRight
 6: WristLeft         19: AnkleRight
 7: HandLeft          20: FootRight
 8: ShoulderRight     21: SpineShoulder
 9: ElbowRight        22: HandTipLeft
10: WristRight        23: ThumbLeft
11: HandRight         24: HandTipRight
12: HipLeft           25: ThumbRight (possibly)
```

**Note:** Test 3 will verify this structure!

## Testing Output

After running `test_stream_conversion.py`, you'll get:

### Console Output
1. Statistics for all 162 files (frame counts, sizes)
2. Sample conversion results
3. Kinect structure analysis
4. Movement pattern analysis

### Files in `./test_conversion_output/`
1. **sample_expert_skel.npy** - Raw converted (T, 350)
2. **sample_reshaped_25x14.npy** - Reshaped (T, 25, 14)
3. **sample_positions_25x3.npy** - Positions only (T, 25, 3)
4. **analysis_summary.json** - Metadata and statistics

## What We Found

✓ **162 skeleton stream files** detected
- 81 expert.skel.stream~
- 81 novice.skel.stream~

✓ **350 dimensions** per frame
- Likely: 25 joints × 14 features (Kinect v2)

✓ **Binary format**
- 4-byte floats (float32)
- Sequential frames

## Integration Options

After conversion, you can:

### Option 1: Use Directly in PyTorch
```python
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, npy_dir):
        self.files = glob.glob(f'{npy_dir}/*.npy')

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.from_numpy(data)
```

### Option 2: Integrate with HDF5 Pipeline
```python
# Add to hdf5_pipeline/data_hdf5_flexible.py
# Store skeleton data alongside OpenFace/ViT features
```

### Option 3: Keep Separate
```python
# Use standalone for skeleton-specific tasks
```

## Comparison: Original vs New

| Aspect | Original Notebook | New Script |
|--------|-------------------|------------|
| **Focus** | All streams (au, face, head, skel) | Skel only (350 dim) |
| **Format** | Binary → CSV → numpy | Binary → numpy |
| **Design** | Procedural | Class-based |
| **Testing** | Manual inspection | Automated tests |
| **Documentation** | Comments | Complete docs |
| **Verification** | Visual | Statistical analysis |

## Next Steps - Discussion Points

### 1. Run the Test
```bash
python test_stream_conversion.py
```
This will verify everything works and show you the data structure.

### 2. Normalization Strategy

**Questions:**
- Should skeleton positions be normalized?
- Center by torso position?
- Scale by person height?
- Use as-is?

**Similar to our ViT discussion:**
- If positions are in meters (physical units) → might need normalization
- If already normalized by Kinect → use as-is
- Test 3 will help determine this!

### 3. Integration

**Options:**
- **Standalone**: Keep skeleton processing separate
- **With HDF5**: Integrate into `hdf5_pipeline/`
- **Multi-modal**: Combine skeleton + OpenFace + ViT

**Which makes sense for your use case?**

### 4. Feature Understanding

After Test 3, we'll know:
- Exact structure of 350 dimensions
- Which features are positions
- Which are orientations
- Movement patterns

### 5. Use Case

**Questions:**
- Will you use skeleton alone or with other modalities?
- What's the prediction task?
- Do you need all 25 joints or subset?
- Frame-level or sequence-level features?

## Dependencies

**Required:**
- Python 3.8+
- numpy (for conversion)
- struct (built-in)
- glob (built-in)

**Optional:**
- tqdm (for progress bars)
- pandas (for original notebook compatibility)

## File Locations

```
/home/s2020425/NoXiRe/
├── data/
│   ├── aria-noxi/           # Input: Binary streams
│   └── processed_skel/      # Output: Numpy arrays
│
└── data_preparation/
    ├── format_conversion.py
    ├── test_stream_conversion.py
    ├── test_conversion_output/  # Test results
    └── [documentation files]
```

## Status

✅ **Ready to Test**
- All scripts created
- 162 stream files detected
- Documentation complete

⏭️ **Next: Run the test!**
```bash
python test_stream_conversion.py
```

## Summary

We've created a clean, focused pipeline for skeleton stream conversion:

1. ✅ Distilled code from notebook (focused on skel.stream~)
2. ✅ Clean class-based design (StreamConverter)
3. ✅ Comprehensive testing (3 tests covering all requirements)
4. ✅ Structure analysis (Kinect v2 hypothesis)
5. ✅ Complete documentation

**The test script addresses all your requirements:**
- ✓ Load all 162 files and prove it
- ✓ Convert one file and save for inspection
- ✓ Analyze 350-dim structure to verify Kinect format

**Now run the test to see your data!** 🚀
