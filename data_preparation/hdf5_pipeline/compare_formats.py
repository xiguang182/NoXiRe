"""
Comparison script between pickle and HDF5 formats.
Demonstrates the advantages of HDF5 for slicing and searching.
"""

import pickle
import h5py
import time
import os
import numpy as np


def compare_file_sizes(pickle_path, hdf5_path):
    """Compare file sizes between pickle and HDF5."""
    print("="*60)
    print("FILE SIZE COMPARISON")
    print("="*60)

    if os.path.exists(pickle_path):
        pickle_size = os.path.getsize(pickle_path) / (1024**2)  # MB
        print(f"Pickle file: {pickle_size:.2f} MB")
    else:
        print(f"Pickle file not found: {pickle_path}")
        pickle_size = None

    if os.path.exists(hdf5_path):
        hdf5_size = os.path.getsize(hdf5_path) / (1024**2)  # MB
        print(f"HDF5 file:   {hdf5_size:.2f} MB")
    else:
        print(f"HDF5 file not found: {hdf5_path}")
        hdf5_size = None

    if pickle_size and hdf5_size:
        compression_ratio = (1 - hdf5_size / pickle_size) * 100
        print(f"\nCompression: {compression_ratio:.1f}% size reduction with HDF5")


def compare_load_all_time(pickle_path, hdf5_path):
    """Compare time to load entire dataset."""
    print("\n" + "="*60)
    print("FULL DATASET LOADING TIME")
    print("="*60)

    # Pickle loading
    if os.path.exists(pickle_path):
        start = time.time()
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        pickle_time = time.time() - start
        print(f"Pickle: {pickle_time:.4f} seconds")
        num_samples_pickle = len(pickle_data)
    else:
        print(f"Pickle file not found: {pickle_path}")
        pickle_time = None
        pickle_data = None

    # HDF5 loading (load all data)
    if os.path.exists(hdf5_path):
        start = time.time()
        hdf5_data = []
        with h5py.File(hdf5_path, 'r') as hf:
            num_samples_hdf5 = hf.attrs['num_samples']
            for idx in range(num_samples_hdf5):
                sample_group = hf[f'sample_{idx:03d}']
                exp_data = {
                    'face': sample_group['expert']['face'][:],
                    'aus': sample_group['expert']['aus'][:],
                    'head': sample_group['expert']['head'][:]
                }
                nov_data = {
                    'face': sample_group['novice']['face'][:],
                    'aus': sample_group['novice']['aus'][:],
                    'head': sample_group['novice']['head'][:]
                }
                hdf5_data.append((exp_data, nov_data))
        hdf5_time = time.time() - start
        print(f"HDF5:   {hdf5_time:.4f} seconds")
    else:
        print(f"HDF5 file not found: {hdf5_path}")
        hdf5_time = None
        hdf5_data = None

    if pickle_time and hdf5_time:
        if pickle_time > hdf5_time:
            speedup = pickle_time / hdf5_time
            print(f"\nHDF5 is {speedup:.2f}x faster for full load")
        else:
            slowdown = hdf5_time / pickle_time
            print(f"\nPickle is {slowdown:.2f}x faster for full load")


def compare_random_access(pickle_path, hdf5_path, num_accesses=10):
    """Compare random access performance."""
    print("\n" + "="*60)
    print(f"RANDOM ACCESS TIME ({num_accesses} accesses)")
    print("="*60)

    # Pickle: Must load entire file first
    if os.path.exists(pickle_path):
        start = time.time()
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        load_time = time.time() - start

        # Then access random samples
        np.random.seed(42)
        indices = np.random.randint(0, len(pickle_data), num_accesses)
        start = time.time()
        for idx in indices:
            _ = pickle_data[idx][0]['face']  # Access expert face data
        access_time = time.time() - start
        pickle_total = load_time + access_time

        print(f"Pickle:")
        print(f"  Load time:   {load_time:.4f} seconds")
        print(f"  Access time: {access_time:.4f} seconds")
        print(f"  Total:       {pickle_total:.4f} seconds")
    else:
        print(f"Pickle file not found: {pickle_path}")
        pickle_total = None

    # HDF5: Direct access without loading all
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as hf:
            num_samples = hf.attrs['num_samples']

        np.random.seed(42)
        indices = np.random.randint(0, num_samples, num_accesses)
        start = time.time()
        with h5py.File(hdf5_path, 'r') as hf:
            for idx in indices:
                _ = hf[f'sample_{idx:03d}']['expert']['face'][:]
        hdf5_total = time.time() - start

        print(f"\nHDF5:")
        print(f"  Total:       {hdf5_total:.4f} seconds")
    else:
        print(f"HDF5 file not found: {hdf5_path}")
        hdf5_total = None

    if pickle_total and hdf5_total:
        speedup = pickle_total / hdf5_total
        print(f"\nHDF5 is {speedup:.2f}x faster for random access")


def compare_slicing(pickle_path, hdf5_path, sample_idx=0):
    """Compare slicing performance (getting subset of frames)."""
    print("\n" + "="*60)
    print(f"FRAME SLICING TIME (frames 100-200 of sample {sample_idx})")
    print("="*60)

    # Pickle: Must load entire sample
    if os.path.exists(pickle_path):
        start = time.time()
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        full_data = pickle_data[sample_idx][0]['face']
        sliced_data = full_data[100:200]
        pickle_time = time.time() - start
        print(f"Pickle: {pickle_time:.4f} seconds (loads entire dataset, then slices)")
        print(f"  Full data shape: {full_data.shape}")
        print(f"  Sliced data shape: {sliced_data.shape}")
    else:
        print(f"Pickle file not found: {pickle_path}")
        pickle_time = None

    # HDF5: Direct slice access
    if os.path.exists(hdf5_path):
        start = time.time()
        with h5py.File(hdf5_path, 'r') as hf:
            sliced_data = hf[f'sample_{sample_idx:03d}']['expert']['face'][100:200]
        hdf5_time = time.time() - start
        print(f"\nHDF5:   {hdf5_time:.4f} seconds (loads only requested frames)")
        print(f"  Sliced data shape: {sliced_data.shape}")
    else:
        print(f"HDF5 file not found: {hdf5_path}")
        hdf5_time = None

    if pickle_time and hdf5_time:
        speedup = pickle_time / hdf5_time
        print(f"\nHDF5 is {speedup:.2f}x faster for slicing")


def demonstrate_hdf5_features(hdf5_path):
    """Demonstrate HDF5-specific features."""
    print("\n" + "="*60)
    print("HDF5-SPECIFIC FEATURES")
    print("="*60)

    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found: {hdf5_path}")
        return

    with h5py.File(hdf5_path, 'r') as hf:
        print("\n1. METADATA ACCESS (without loading data):")
        print(f"   Number of samples: {hf.attrs['num_samples']}")
        print(f"   Description: {hf.attrs['description']}")

        print("\n2. SAMPLE INFORMATION:")
        for idx in range(min(3, hf.attrs['num_samples'])):
            sample = hf[f'sample_{idx:03d}']
            print(f"   Sample {idx}:")
            print(f"     Name: {sample.attrs['sample_name']}")
            print(f"     Expert frames: {sample.attrs['num_frames_expert']}")
            print(f"     Novice frames: {sample.attrs['num_frames_novice']}")

        print("\n3. SELECTIVE FEATURE LOADING:")
        print("   Loading only 'aus' data from sample 0 (expert)...")
        start = time.time()
        aus_only = hf['sample_000']['expert']['aus'][:]
        load_time = time.time() - start
        print(f"   Loaded {aus_only.shape} in {load_time:.4f} seconds")
        print("   (No need to load face or head data!)")

        print("\n4. MEMORY-EFFICIENT ITERATION:")
        print("   Processing samples without loading all into memory...")
        start = time.time()
        mean_aus_values = []
        for idx in range(hf.attrs['num_samples']):
            aus = hf[f'sample_{idx:03d}']['expert']['aus'][:]
            mean_aus_values.append(np.mean(aus, axis=0))
        iter_time = time.time() - start
        print(f"   Processed {hf.attrs['num_samples']} samples in {iter_time:.4f} seconds")
        print(f"   (Only one sample in memory at a time)")


def main():
    pickle_path = './data/test.pkl'
    hdf5_path = './data/openface_data.h5'

    print("\n" + "="*60)
    print("PICKLE vs HDF5 FORMAT COMPARISON")
    print("="*60)

    compare_file_sizes(pickle_path, hdf5_path)
    compare_load_all_time(pickle_path, hdf5_path)
    compare_random_access(pickle_path, hdf5_path, num_accesses=10)
    compare_slicing(pickle_path, hdf5_path, sample_idx=0)
    demonstrate_hdf5_features(hdf5_path)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nHDF5 advantages:")
    print("  ✓ Smaller file size (compression)")
    print("  ✓ Faster random access (no need to load entire dataset)")
    print("  ✓ Efficient slicing (load only needed frames)")
    print("  ✓ Metadata support (sample names, frame counts, etc.)")
    print("  ✓ Memory efficient (process one sample at a time)")
    print("  ✓ Selective feature loading (load only face/aus/head as needed)")
    print("\nPickle advantages:")
    print("  ✓ Simpler for small datasets")
    print("  ✓ Can store arbitrary Python objects")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
