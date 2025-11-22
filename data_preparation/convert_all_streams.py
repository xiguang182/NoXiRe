"""
Convert all skeleton streams to numpy arrays.

This script uses the StreamConverter to convert all 162 skel.stream~ files
from binary format to numpy arrays and saves them to data/processed_data.

Usage:
    python convert_all_streams.py
"""

import os
import sys
from pathlib import Path
from format_conversion import StreamConverter


def main():
    """Convert all skeleton streams and save to processed_data."""

    print("=" * 70)
    print("NoXiRe Skeleton Stream Conversion")
    print("=" * 70)

    # Setup paths
    root_dir = '../data/aria-noxi'
    save_dir = '../data/processed_data/skeleton'

    # Check if input directory exists
    if not os.path.exists(root_dir):
        print(f"\n✗ Error: Input directory not found: {root_dir}")
        print(f"  Please ensure the aria-noxi data is at: {os.path.abspath(root_dir)}")
        sys.exit(1)

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nInput directory: {os.path.abspath(root_dir)}")
    print(f"Output directory: {os.path.abspath(save_dir)}")

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
        print("  Expected pattern: */expert.skel.stream~ or */novice.skel.stream~")
        sys.exit(1)

    print(f"✓ Found {len(stream_files)} skeleton stream files")

    # Confirm before proceeding
    print("\nThis will convert all files to numpy arrays.")
    response = input("Proceed? (y/n): ")

    if response.lower() != 'y':
        print("Conversion cancelled.")
        sys.exit(0)

    # Convert all streams
    print("\nConverting streams...")
    print("-" * 70)

    results = converter.convert_all_streams(
        pattern='*skel.stream~',
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    if results:
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"\nTotal files processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")

        if successful > 0:
            print(f"\n✓ Output saved to: {os.path.abspath(save_dir)}")

            # Show data structure info
            print("\nOutput file structure:")
            print("  Format: {sample}_{person}_skel.npy")
            print("  Shape: (T, 350) where T = number of frames")
            print("  Structure: 25 joints × 14 features per joint")
            print("    - Features 0-2: Position (x, y, z) in meters")
            print("    - Features 3-6: Orientation quaternion (x, y, z, w)")
            print("    - Features 7-13: Tracking metadata")

        # Show any failures
        if failed > 0:
            print(f"\n⚠ {failed} file(s) failed to convert:")
            for result in results:
                if not result['success']:
                    print(f"  ✗ {result['sample']} ({result['person']}): {result.get('error', 'Unknown error')}")
    else:
        print("\n✗ No files were converted")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
