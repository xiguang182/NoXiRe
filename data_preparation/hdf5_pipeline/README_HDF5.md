# HDF5 Data Format Guide

This guide explains how to use the HDF5 format for the NoXiRe dataset, which provides better performance for slicing and searching compared to pickle format.

## Quick Start

### 1. Convert your data to HDF5

```python
python data_hdf5.py
```

This will create `./data/openface_data.h5` with all your OpenFace data.

### 2. Compare performance

```python
python compare_formats.py
```

This will show the performance differences between pickle and HDF5.

## Data Structure

The HDF5 file is organized as follows:

```
openface_data.h5
├── sample_names (dataset with all sample names)
├── sample_000/
│   ├── expert/
│   │   ├── face: (T, 68, 2) - facial landmarks
│   │   ├── aus: (T, 17) - action units
│   │   └── head: (T, 6) - head pose
│   └── novice/
│       ├── face: (T, 68, 2)
│       ├── aus: (T, 17)
│       └── head: (T, 6)
├── sample_001/
│   └── ...
```

Where `T` is the number of frames (varies per sample).

## Usage Examples

### Basic Loading

```python
import h5py
from data_hdf5 import load_sample, load_slice, search_samples_by_frame_count

# Load entire sample
data = load_sample('./data/openface_data.h5', sample_idx=0, person='expert')
print(data['face'].shape)  # (T, 68, 2)
print(data['aus'].shape)   # (T, 17)
print(data['head'].shape)  # (T, 6)

# Load by name
data = load_sample_by_name('./data/openface_data.h5',
                          sample_name='001.001.001.001',
                          person='novice')
```

### Efficient Slicing

```python
# Load only frames 100-200 (without loading entire sample!)
face_subset = load_slice('./data/openface_data.h5',
                        sample_idx=0,
                        person='expert',
                        feature='face',
                        frame_start=100,
                        frame_end=200)
print(face_subset.shape)  # (100, 68, 2)

# Load only action units for all frames
aus = load_slice('./data/openface_data.h5',
                sample_idx=5,
                person='expert',
                feature='aus')
```

### Searching

```python
# Find samples with 100-500 frames
samples = search_samples_by_frame_count('./data/openface_data.h5',
                                       person='expert',
                                       min_frames=100,
                                       max_frames=500)

for idx, name, num_frames in samples:
    print(f"Sample {idx} ({name}): {num_frames} frames")
```

### Memory-Efficient Batch Processing

```python
import h5py
import numpy as np

# Process all samples without loading everything into memory
with h5py.File('./data/openface_data.h5', 'r') as hf:
    num_samples = hf.attrs['num_samples']

    results = []
    for idx in range(num_samples):
        # Only one sample in memory at a time
        expert_aus = hf[f'sample_{idx:03d}']['expert']['aus'][:]

        # Compute statistics
        mean_au = np.mean(expert_aus, axis=0)
        results.append(mean_au)

    # Aggregate results
    overall_mean = np.mean(results, axis=0)
```

### Dataset Information

```python
from data_hdf5 import get_dataset_info

# Print comprehensive dataset info
get_dataset_info('./data/openface_data.h5')
```

### Advanced: Direct HDF5 Access

```python
import h5py

with h5py.File('./data/openface_data.h5', 'r') as hf:
    # Access metadata
    num_samples = hf.attrs['num_samples']

    # Iterate over samples
    for idx in range(num_samples):
        sample = hf[f'sample_{idx:03d}']
        sample_name = sample.attrs['sample_name']

        # Access specific features
        expert_face = sample['expert']['face']

        # Get shape without loading data
        print(f"{sample_name}: {expert_face.shape}")

        # Load specific frames
        first_10_frames = expert_face[:10]

        # Load specific landmarks
        nose_landmark = expert_face[:, 30, :]  # 30th landmark
```

## PyTorch DataLoader Integration

```python
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

class NoXiReHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, person='expert', feature='face'):
        self.hdf5_path = hdf5_path
        self.person = person
        self.feature = feature

        # Load metadata only
        with h5py.File(hdf5_path, 'r') as hf:
            self.num_samples = hf.attrs['num_samples']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as hf:
            data = hf[f'sample_{idx:03d}'][self.person][self.feature][:]
        return torch.from_numpy(data)

# Usage
dataset = NoXiReHDF5Dataset('./data/openface_data.h5',
                           person='expert',
                           feature='face')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for batch in dataloader:
    # batch shape: (batch_size, T, 68, 2) for face data
    print(batch.shape)
    break
```

## Performance Benefits

### 1. File Size
- HDF5 with compression is typically 40-60% smaller than pickle
- `compression='gzip', compression_opts=4` provides good balance

### 2. Random Access
- Pickle: Must load entire file (~2-5 seconds for large datasets)
- HDF5: Direct access to any sample (~0.01-0.05 seconds)

### 3. Slicing
- Pickle: Load full sample, then slice in memory
- HDF5: Load only requested frames directly from disk

### 4. Memory Efficiency
- Process samples one at a time without loading entire dataset
- Essential for large datasets that don't fit in RAM

### 5. Metadata
- Store and query sample information without loading data
- Useful for filtering and searching

## Migration from Pickle

If you have existing code using pickle format:

```python
# Old pickle code
with open('./data/test.pkl', 'rb') as f:
    data = pickle.load(f)

expert_face = data[0][0]['face']
novice_aus = data[0][1]['aus']
```

Equivalent HDF5 code:

```python
# New HDF5 code
from data_hdf5 import load_sample

data_expert = load_sample('./data/openface_data.h5', 0, 'expert')
data_novice = load_sample('./data/openface_data.h5', 0, 'novice')

expert_face = data_expert['face']
novice_aus = data_novice['aus']
```

## Best Practices

1. **Keep file handle open for multiple accesses**
   ```python
   with h5py.File(path, 'r') as hf:
       # Multiple accesses here
       data1 = hf['sample_000']['expert']['face'][:]
       data2 = hf['sample_001']['expert']['face'][:]
   ```

2. **Use slicing to reduce memory usage**
   ```python
   # Good: Load only what you need
   subset = hf['sample_000']['expert']['face'][100:200]

   # Avoid: Loading full data then slicing
   full = hf['sample_000']['expert']['face'][:]
   subset = full[100:200]
   ```

3. **Check shapes before loading**
   ```python
   face_data = hf['sample_000']['expert']['face']
   print(face_data.shape)  # Doesn't load data
   actual_data = face_data[:]  # Now load
   ```

4. **Use context managers**
   ```python
   # Always use 'with' to ensure file is closed
   with h5py.File(path, 'r') as hf:
       data = hf['sample_000']['expert']['face'][:]
   ```

## Troubleshooting

### Issue: "Unable to open file"
- Make sure the HDF5 file exists: run `data_hdf5.py` first
- Check file permissions

### Issue: Slow performance
- Use compression for storage, but note it adds CPU overhead
- For frequently accessed data, consider loading into memory once
- Use `num_workers > 0` in PyTorch DataLoader for parallel loading

### Issue: "Dataset not found"
- Check sample indices (0-based)
- Verify person is 'expert' or 'novice'
- Verify feature is 'face', 'aus', or 'head'

## Dependencies

```bash
pip install h5py numpy pandas tqdm
```

## References

- [HDF5 Documentation](https://docs.h5py.org/)
- [HDF5 Best Practices](https://docs.h5py.org/en/stable/high/dataset.html)
