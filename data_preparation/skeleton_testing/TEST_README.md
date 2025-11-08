# Stream Conversion Testing

## Overview

The `test_stream_conversion.py` script performs comprehensive testing of the skeleton stream conversion pipeline.

## What It Tests

### Test 1: Load All Stream Files ✓
- Finds all 162 skel.stream~ files
- Reads each file and calculates frame count
- Verifies file integrity (checks for incomplete files)
- Prints statistics: min/max/mean frames, file sizes

### Test 2: Convert Single File ✓
- Converts one sample file (001_2016-03-17_Paris/expert.skel.stream~)
- Saves to `./test_conversion_output/` folder
- Prints data statistics and sample values
- Verifies the conversion worked correctly

### Test 3: Analyze Kinect Structure ✓
- Analyzes the 350-dimension structure
- Tests hypothesis: **25 joints × 14 features**
- Identifies position coordinates (x, y, z)
- Checks for quaternion orientation
- Analyzes movement patterns
- Saves analysis results

## Expected Kinect v2 Structure

```
350 dimensions = 25 joints × 14 features

Per joint (14 features):
  [0-2]:   Position (x, y, z) in meters
  [3-6]:   Orientation quaternion (x, y, z, w)?
  [7-13]:  Other features (confidence, tracking state, etc.)?
```

## Usage

```bash
cd /home/s2020425/NoXiRe/data_preparation
python test_stream_conversion.py
```

**Note:** Requires numpy to be installed in your environment.

## Output

The script creates `./test_conversion_output/` with:

1. **sample_expert_skel.npy** - Converted numpy array (T, 350)
2. **sample_reshaped_25x14.npy** - Reshaped to (T, 25, 14)
3. **sample_positions_25x3.npy** - Just positions (T, 25, 3)
4. **analysis_summary.json** - Statistics and metadata

## What You'll Learn

After running this test, you'll know:

- ✓ How many skeleton stream files you have
- ✓ Frame count for each file
- ✓ Whether files are complete or corrupted
- ✓ The structure of the 350 dimensions
- ✓ Which features are positions
- ✓ Movement patterns in the data
- ✓ Kinect joint mapping

## Kinect v2 Joint Names

If confirmed to be Kinect v2, the 25 joints are:

```
 0: SpineBase        13: HipLeft
 1: SpineMid         14: KneeLeft
 2: Neck             15: AnkleLeft
 3: Head             16: FootLeft
 4: ShoulderLeft     17: HipRight
 5: ElbowLeft        18: KneeRight
 6: WristLeft        19: AnkleRight
 7: HandLeft         20: FootRight
 8: ShoulderRight    21: SpineShoulder
 9: ElbowRight       22: HandTipLeft
10: WristRight       23: ThumbLeft
11: HandRight        24: HandTipRight
12: HipLeft          25: ThumbRight (if 26 joints, else omitted)
```

## Next Steps

After running the test:

1. **Review the output** to understand the data structure
2. **Check analysis_summary.json** for key findings
3. **Decide on normalization** strategy for skeleton data
4. **Integrate with HDF5 pipeline** if needed

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
- You need numpy installed in your Python environment
- The test requires numpy for array operations

### "File not found"
- Make sure you're running from `/home/s2020425/NoXiRe/data_preparation/`
- Check that `../data/aria-noxi/` exists

### "No files found"
- Verify the aria-noxi directory structure
- Check that .skel.stream~ files exist
