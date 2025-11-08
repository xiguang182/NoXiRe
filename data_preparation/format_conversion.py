"""
Binary Stream to Numpy Conversion for NoXiRe Dataset

This script converts binary skel.stream~ files (350-dimensional skeleton data)
into numpy arrays for efficient processing.

File structure expected:
    data/aria-noxi/
    ├── 001_2016-03-17_Paris/
    │   ├── expert.skel.stream~
    │   └── novice.skel.stream~
    └── [other samples...]

Output:
    Numpy arrays saved as .npy files with skeleton features (T, 350)
"""

import numpy as np
import pandas as pd
import os
import glob
import struct
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional


class StreamConverter:
    """Convert binary stream files to numpy arrays."""

    def __init__(self, root_dir: str = '../data/aria-noxi',
                 save_dir: str = '../data/processed_skel',
                 feature_dim: int = 350):
        """
        Initialize converter.

        Args:
            root_dir: Root directory containing aria-noxi data
            save_dir: Directory to save processed numpy files
            feature_dim: Dimension of skeleton features (default: 350)
        """
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.feature_dim = feature_dim
        self.byte_size = 4  # float32

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def search_directories(self) -> List[str]:
        """Find all sample directories in root."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        dirs = [f for f in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, f))]
        return sorted(dirs)

    def search_stream_files(self, pattern: str = '*skel.stream~') -> List[str]:
        """
        Search for stream files matching pattern.

        Args:
            pattern: Glob pattern to match files (default: '*skel.stream~')

        Returns:
            List of full paths to stream files
        """
        all_streams = []
        samples = self.search_directories()

        for sample in samples:
            stream_list = glob.glob(os.path.join(self.root_dir, sample, pattern))
            stream_list.sort()
            all_streams.extend(stream_list)

        return all_streams

    def binary_to_numpy(self, stream_path: str, verbose: bool = False) -> np.ndarray:
        """
        Convert binary stream file to numpy array.

        Args:
            stream_path: Path to binary stream file
            verbose: Print warnings if incomplete samples found

        Returns:
            Numpy array of shape (T, feature_dim) where T is number of frames
        """
        feat_list = []
        bytes_per_sample = self.feature_dim * self.byte_size

        with open(stream_path, 'rb') as f:
            while True:
                # Read one complete feature sample
                val = f.read(bytes_per_sample)

                # Check if we've reached end of file
                if len(val) == 0:
                    break

                # Check for incomplete sample (corrupted file)
                if len(val) < bytes_per_sample:
                    if verbose:
                        print(f'Warning: Incomplete feature sample in {stream_path}')
                        print(f'  Expected {bytes_per_sample} bytes, got {len(val)} bytes')
                    break

                # Unpack floats from binary
                feat_set = []
                for i in range(0, bytes_per_sample, self.byte_size):
                    num = struct.unpack('f', val[i:i+self.byte_size])[0]
                    feat_set.append(num)

                feat_list.append(feat_set)

        # Convert to numpy array
        return np.array(feat_list, dtype=np.float32)

    def get_save_path(self, stream_path: str, format: str = 'npy') -> str:
        """
        Generate save path for processed file.

        Args:
            stream_path: Original stream file path
            format: Output format ('npy' or 'csv')

        Returns:
            Path to save processed file

        Example:
            Input:  './data/aria-noxi/001_2016-03-17_Paris/expert.skel.stream~'
            Output: './data/processed_skel/001_2016-03-17_Paris_expert_skel.npy'
        """
        # Get sample directory name
        sample_dir = Path(stream_path).parent.name

        # Get filename components
        filename = Path(stream_path).name
        parts = filename.replace('.stream~', '').split('.')

        # Build output filename: sample_person_feature.npy
        # E.g., 001_2016-03-17_Paris_expert_skel.npy
        output_name = f"{sample_dir}_{'_'.join(parts)}.{format}"

        return os.path.join(self.save_dir, output_name)

    def convert_stream_to_numpy(self, stream_path: str,
                                save: bool = True,
                                verbose: bool = False) -> Tuple[np.ndarray, str]:
        """
        Convert single stream file to numpy and optionally save.

        Args:
            stream_path: Path to stream file
            save: Whether to save the numpy array
            verbose: Print conversion info

        Returns:
            Tuple of (numpy_array, save_path)
        """
        # Convert to numpy
        data = self.binary_to_numpy(stream_path, verbose=verbose)

        # Generate save path
        save_path = self.get_save_path(stream_path, format='npy')

        # Save if requested
        if save:
            np.save(save_path, data)
            if verbose:
                print(f"Saved: {save_path} with shape {data.shape}")

        return data, save_path

    def convert_all_streams(self, pattern: str = '*skel.stream~',
                           verbose: bool = True) -> List[Tuple[str, np.ndarray]]:
        """
        Convert all stream files matching pattern.

        Args:
            pattern: Glob pattern for files to convert
            verbose: Show progress bar and info

        Returns:
            List of (save_path, array) tuples
        """
        stream_files = self.search_stream_files(pattern)

        if len(stream_files) == 0:
            print(f"No files found matching pattern: {pattern}")
            return []

        print(f"Found {len(stream_files)} stream files to convert")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Output directory: {self.save_dir}")

        results = []
        iterator = tqdm(stream_files, desc='Converting streams') if verbose else stream_files

        for stream_path in iterator:
            try:
                data, save_path = self.convert_stream_to_numpy(
                    stream_path,
                    save=True,
                    verbose=False
                )
                results.append((save_path, data))
            except Exception as e:
                print(f"\nError converting {stream_path}: {e}")
                continue

        if verbose:
            print(f"\nSuccessfully converted {len(results)} files")
            if len(results) > 0:
                sample_shape = results[0][1].shape
                print(f"Sample shape: {sample_shape} (frames × features)")

        return results

    def verify_conversion(self, sample_idx: int = 0) -> None:
        """
        Load and verify a converted file.

        Args:
            sample_idx: Index of sample to verify
        """
        npy_files = sorted(glob.glob(os.path.join(self.save_dir, '*.npy')))

        if len(npy_files) == 0:
            print("No .npy files found to verify")
            return

        if sample_idx >= len(npy_files):
            print(f"Sample index {sample_idx} out of range. Max: {len(npy_files)-1}")
            return

        file_path = npy_files[sample_idx]
        data = np.load(file_path)

        print(f"\nVerification of: {Path(file_path).name}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Memory: {data.nbytes / 1024 / 1024:.2f} MB")
        print(f"\nStatistics:")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std:  {data.std():.4f}")
        print(f"  Min:  {data.min():.4f}")
        print(f"  Max:  {data.max():.4f}")
        print(f"\nFirst 3 frames, first 5 features:")
        print(data[:3, :5])

    def get_conversion_summary(self) -> dict:
        """Get summary of converted files."""
        npy_files = glob.glob(os.path.join(self.save_dir, '*.npy'))

        if len(npy_files) == 0:
            return {'num_files': 0}

        # Load first file to get shape info
        sample_data = np.load(npy_files[0])

        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in npy_files)

        return {
            'num_files': len(npy_files),
            'feature_dim': sample_data.shape[1],
            'sample_shape': sample_data.shape,
            'total_size_mb': total_size / 1024 / 1024,
            'avg_size_mb': (total_size / len(npy_files)) / 1024 / 1024,
            'output_dir': self.save_dir
        }


def main():
    """Main conversion pipeline."""

    print("="*60)
    print("NoXiRe Skeleton Stream Conversion")
    print("="*60)

    # Initialize converter
    converter = StreamConverter(
        root_dir='../data/aria-noxi',
        save_dir='../data/processed_skel',
        feature_dim=350
    )

    # Convert all skel.stream~ files
    results = converter.convert_all_streams(pattern='*skel.stream~', verbose=True)

    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)

    summary = converter.get_conversion_summary()
    if summary['num_files'] > 0:
        print(f"Files converted: {summary['num_files']}")
        print(f"Feature dimension: {summary['feature_dim']}")
        print(f"Sample shape: {summary['sample_shape']}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB")
        print(f"Avg size per file: {summary['avg_size_mb']:.2f} MB")
        print(f"Output directory: {summary['output_dir']}")

        # Verify first file
        print("\n" + "="*60)
        print("Sample Verification")
        print("="*60)
        converter.verify_conversion(sample_idx=0)
    else:
        print("No files were converted")


if __name__ == '__main__':
    main()
