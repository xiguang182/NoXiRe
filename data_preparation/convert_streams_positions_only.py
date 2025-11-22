"""
Convert all skeleton streams to numpy arrays - POSITIONS ONLY (75 dimensions).

This script converts skeleton streams and extracts only the 3D joint positions,
reducing from 350 dims to 75 dims (25 joints × 3 coordinates).

Output: (T, 75) arrays with flattened x,y,z positions for all 25 joints

Usage:
    python convert_streams_positions_only.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from format_conversion import StreamConverter


def extract_positions(data: np.ndarray) -> np.ndarray:
    """
    Extract only position features from skeleton data.

    Args:
        data: (T, 350) array with full skeleton features

    Returns:
        positions: (T, 75) array with positions only
    """
    # Reshape to (T, 25 joints, 14 features)
    T = data.shape[0]
    reshaped = data.reshape(T, 25, 14)

    # Extract first 3 features per joint (x, y, z positions)
    positions = reshaped[:, :, :3]  # (T, 25, 3)

    # Flatten back to (T, 75)
    positions_flat = positions.reshape(T, 75)

    return positions_flat


def main():
    """Convert all skeleton streams and save positions only."""

    print("=" * 70)
    print("NoXiRe Skeleton Stream Conversion - POSITIONS ONLY")
    print("=" * 70)

    # Setup paths
    root_dir = '../data/aria-noxi'
    save_dir = '../data/processed_data/skeleton_positions'

    # Check if input directory exists
    if not os.path.exists(root_dir):
        print(f"\n✗ Error: Input directory not found: {root_dir}")
        print(f"  Please ensure the aria-noxi data is at: {os.path.abspath(root_dir)}")
        sys.exit(1)

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nInput directory: {os.path.abspath(root_dir)}")
    print(f"Output directory: {os.path.abspath(save_dir)}")
    print(f"\nExtraction mode: POSITIONS ONLY (75 dims)")
    print("  - 25 joints × 3 coordinates (x, y, z)")
    print("  - Discards orientation and metadata")

    # Initialize converter
    converter = StreamConverter(
        root_dir=root_dir,
        save_dir=save_dir,
        feature_dim=350
    )

    # Find all stream files
    print("\nSearching for skeleton stream files...")
    stream_files = converter.search_stream_files(pattern='*skel.stream~')

    if len(stream_files) == 0:
        print(f"\n✗ Error: No skeleton stream files found in {root_dir}")
        sys.exit(1)

    print(f"✓ Found {len(stream_files)} skeleton stream files")

    # Confirm before proceeding
    print("\nThis will convert all files to position-only numpy arrays (75 dims).")
    response = input("Proceed? (y/n): ")

    if response.lower() != 'y':
        print("Conversion cancelled.")
        sys.exit(0)

    # Convert all streams
    print("\nConverting streams and extracting positions...")
    print("-" * 70)

    successful = 0
    failed = 0

    for stream_path in stream_files:
        # Extract sample name and person from path
        path_parts = Path(stream_path).parts
        sample = path_parts[-2]
        filename = path_parts[-1]
        person = 'expert' if 'expert' in filename else 'novice'

        try:
            # Convert to numpy (full 350 dims)
            data = converter.binary_to_numpy(stream_path, verbose=False)

            # Extract positions only (75 dims)
            positions = extract_positions(data)

            # Save
            output_filename = f"{sample}_{person}_positions.npy"
            output_path = os.path.join(save_dir, output_filename)
            np.save(output_path, positions)

            print(f"✓ {sample} ({person}): {positions.shape} → {output_filename}")
            successful += 1

        except Exception as e:
            print(f"✗ {sample} ({person}): {e}")
            failed += 1

    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"\nTotal files processed: {successful + failed}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if successful > 0:
        print(f"\n✓ Output saved to: {os.path.abspath(save_dir)}")

        print("\nOutput file structure:")
        print("  Format: {sample}_{person}_positions.npy")
        print("  Shape: (T, 75) where T = number of frames")
        print("  Data: 25 joints × 3 coords (x, y, z) in meters")
        print("\nJoint ordering (Kinect v2):")
        print("  0-2:   SpineBase (x,y,z)")
        print("  3-5:   SpineMid (x,y,z)")
        print("  6-8:   Neck (x,y,z)")
        print("  9-11:  Head (x,y,z)")
        print("  ... (25 joints total)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
