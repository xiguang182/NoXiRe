"""
Testing script for skeleton stream conversion.

Tests:
1. Load all 162 stream files and verify they can be read
2. Convert one sample file and save for inspection
3. Analyze the 350-dim structure to understand Kinect skeleton format
"""

import os
import sys
import glob
import struct
import numpy as np
from pathlib import Path

# Add parent directory to path to import format_conversion
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from format_conversion import StreamConverter


def test_1_load_all_streams():
    """
    Test 1: Load all 162 stream files and print basic info.
    Proves all files are readable and shows their sizes.
    """
    print("="*70)
    print("TEST 1: Load All Stream Files")
    print("="*70)

    root_dir = '../../data/aria-noxi'
    pattern = '*/expert.skel.stream~'

    # Find all expert streams
    expert_streams = glob.glob(os.path.join(root_dir, pattern))
    novice_streams = glob.glob(os.path.join(root_dir, '*/novice.skel.stream~'))

    all_streams = sorted(expert_streams + novice_streams)

    print(f"\nFound {len(all_streams)} skeleton stream files")
    print(f"  Expert: {len(expert_streams)}")
    print(f"  Novice: {len(novice_streams)}")

    # Read each file and collect stats
    file_stats = []
    feature_dim = 350
    byte_size = 4  # float32
    bytes_per_frame = feature_dim * byte_size

    print("\nReading all files...")
    for i, stream_path in enumerate(all_streams):
        try:
            # Get file size
            file_size = os.path.getsize(stream_path)

            # Calculate expected number of frames
            num_frames = file_size // bytes_per_frame
            remainder = file_size % bytes_per_frame

            # Extract sample name and person
            path_parts = Path(stream_path).parts
            sample = path_parts[-2]
            filename = path_parts[-1]
            person = 'expert' if 'expert' in filename else 'novice'

            file_stats.append({
                'path': stream_path,
                'sample': sample,
                'person': person,
                'file_size_mb': file_size / (1024**2),
                'num_frames': num_frames,
                'remainder_bytes': remainder,
                'complete': remainder == 0
            })

        except Exception as e:
            print(f"  ✗ Error reading {stream_path}: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print("FILE STATISTICS")
    print(f"{'='*70}")
    print(f"Total files successfully read: {len(file_stats)}")
    print(f"Complete files (no remainder): {sum(1 for s in file_stats if s['complete'])}")
    print(f"Incomplete files: {sum(1 for s in file_stats if not s['complete'])}")

    # Calculate statistics
    frame_counts = [s['num_frames'] for s in file_stats]
    file_sizes = [s['file_size_mb'] for s in file_stats]

    print(f"\nFrame count statistics:")
    print(f"  Min frames: {min(frame_counts):,}")
    print(f"  Max frames: {max(frame_counts):,}")
    print(f"  Mean frames: {np.mean(frame_counts):,.1f}")
    print(f"  Median frames: {np.median(frame_counts):,.1f}")

    print(f"\nFile size statistics:")
    print(f"  Min size: {min(file_sizes):.2f} MB")
    print(f"  Max size: {max(file_sizes):.2f} MB")
    print(f"  Mean size: {np.mean(file_sizes):.2f} MB")
    print(f"  Total size: {sum(file_sizes):.2f} MB")

    # Show first 10 files
    print(f"\n{'='*70}")
    print("FIRST 10 FILES")
    print(f"{'='*70}")
    print(f"{'Sample':<30} {'Person':<10} {'Frames':<10} {'Size (MB)':<10} {'OK'}")
    print("-"*70)
    for stat in file_stats[:10]:
        status = '✓' if stat['complete'] else '✗'
        print(f"{stat['sample']:<30} {stat['person']:<10} {stat['num_frames']:<10,} "
              f"{stat['file_size_mb']:<10.2f} {status}")

    # Show any incomplete files
    incomplete = [s for s in file_stats if not s['complete']]
    if incomplete:
        print(f"\n{'='*70}")
        print("INCOMPLETE FILES (WARNING)")
        print(f"{'='*70}")
        for stat in incomplete:
            print(f"{stat['sample']} ({stat['person']}): {stat['remainder_bytes']} extra bytes")

    return file_stats


def test_2_convert_single_file():
    """
    Test 2: Convert one file and save for testing.
    Creates a test output folder with one converted file.
    """
    print("\n" + "="*70)
    print("TEST 2: Convert Single File")
    print("="*70)

    # Choose first expert file
    test_file = '../../data/aria-noxi/001_2016-03-17_Paris/expert.skel.stream~'
    test_output_dir = './test_conversion_output'

    # Create test output directory
    os.makedirs(test_output_dir, exist_ok=True)

    print(f"\nTest file: {test_file}")
    print(f"Output directory: {test_output_dir}")

    # Check if file exists
    if not os.path.exists(test_file):
        print(f"✗ File not found: {test_file}")
        return None

    # Initialize converter
    converter = StreamConverter(
        root_dir='../../data/aria-noxi',
        save_dir=test_output_dir,
        feature_dim=350
    )

    # Convert the file
    print("\nConverting...")
    try:
        data, save_path = converter.convert_stream_to_numpy(
            test_file,
            save=True,
            verbose=True
        )

        print(f"\n✓ Conversion successful!")
        print(f"  Output: {save_path}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Memory: {data.nbytes / (1024**2):.2f} MB")

        # Basic statistics
        print(f"\nData statistics:")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")

        # Show first frame
        print(f"\nFirst frame (first 10 values):")
        print(data[0, :10])

        # Show last frame
        print(f"\nLast frame (first 10 values):")
        print(data[-1, :10])

        return data

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_3_analyze_kinect_structure(data):
    """
    Test 3: Analyze 350-dim structure to understand Kinect skeleton format.

    Kinect v2 skeleton typically has:
    - 25 joints × 14 features = 350 dimensions

    Each joint may have:
    - Position: (x, y, z) = 3 values
    - Orientation: (x, y, z, w) quaternion = 4 values
    - Tracking state: 1 value
    - Or other configurations

    Let's analyze to figure out the structure.
    """
    print("\n" + "="*70)
    print("TEST 3: Analyze Kinect Skeleton Structure (350 dimensions)")
    print("="*70)

    if data is None:
        print("✗ No data to analyze (conversion failed)")
        return

    print(f"\nData shape: {data.shape}")
    print(f"Total dimensions: {data.shape[1]}")

    # Common Kinect configurations
    print("\n" + "-"*70)
    print("POSSIBLE KINECT CONFIGURATIONS:")
    print("-"*70)

    configs = [
        (25, 14, "25 joints × 14 features (Kinect v2 full)"),
        (25, 11, "25 joints × 11 features"),
        (20, 17.5, "20 joints × 17.5 features"),
        (50, 7, "50 joints × 7 features"),
    ]

    for num_joints, features_per_joint, description in configs:
        if num_joints * features_per_joint == 350:
            print(f"✓ {description}")
        else:
            print(f"  {description} = {num_joints * features_per_joint}")

    # Most likely: 25 joints × 14 features (Kinect v2)
    print("\n" + "-"*70)
    print("ANALYSIS: Assuming 25 joints × 14 features")
    print("-"*70)

    num_joints = 25
    features_per_joint = 14

    # Reshape to see per-joint structure
    # Shape: (num_frames, num_joints, features_per_joint)
    try:
        reshaped = data.reshape(data.shape[0], num_joints, features_per_joint)
        print(f"\nReshaped to: {reshaped.shape} (frames, joints, features)")

        # Analyze first frame, first joint
        print(f"\nFirst frame, first joint (14 features):")
        first_joint = reshaped[0, 0, :]
        for i, val in enumerate(first_joint):
            print(f"  Feature {i:2d}: {val:10.6f}")

        # Check if first 3 values look like positions
        print("\n" + "-"*70)
        print("POSITION ANALYSIS (first 3 features per joint)")
        print("-"*70)

        # Extract first 3 features (likely x, y, z positions)
        positions = reshaped[:, :, :3]  # (frames, joints, 3)

        print(f"\nPosition shape: {positions.shape}")
        print(f"Position statistics across all frames and joints:")
        print(f"  X range: [{positions[:, :, 0].min():.4f}, {positions[:, :, 0].max():.4f}]")
        print(f"  Y range: [{positions[:, :, 1].min():.4f}, {positions[:, :, 1].max():.4f}]")
        print(f"  Z range: [{positions[:, :, 2].min():.4f}, {positions[:, :, 2].max():.4f}]")

        # Check for typical Kinect coordinate ranges
        # Kinect typically uses meters: X [-2, 2], Y [-2, 2], Z [0.5, 4.5]
        x_typical = -3 < positions[:, :, 0].mean() < 3
        y_typical = -3 < positions[:, :, 1].mean() < 3
        z_typical = 0 < positions[:, :, 2].mean() < 5

        if x_typical and y_typical and z_typical:
            print("\n✓ First 3 features appear to be 3D positions (x, y, z) in meters")
        else:
            print("\n⚠ Position ranges seem unusual for Kinect coordinates")

        # Analyze features 4-13
        print("\n" + "-"*70)
        print("OTHER FEATURES ANALYSIS (features 3-13)")
        print("-"*70)

        other_features = reshaped[:, :, 3:]  # Features 3 onwards
        print(f"\nOther features shape: {other_features.shape}")

        for feat_idx in range(min(11, other_features.shape[2])):
            feat_data = other_features[:, :, feat_idx]
            print(f"Feature {feat_idx+3:2d}: "
                  f"mean={feat_data.mean():8.4f}, "
                  f"std={feat_data.std():8.4f}, "
                  f"range=[{feat_data.min():8.4f}, {feat_data.max():8.4f}]")

        # Check if features 3-6 look like quaternion (orientation)
        if other_features.shape[2] >= 4:
            quat = other_features[:, :, :4]  # Features 3-6
            quat_norms = np.linalg.norm(quat, axis=2)
            print(f"\nQuaternion norm check (features 3-6):")
            print(f"  Mean norm: {quat_norms.mean():.4f}")
            print(f"  Std norm: {quat_norms.std():.4f}")
            if 0.9 < quat_norms.mean() < 1.1:
                print("  ✓ Features 3-6 might be quaternion orientation (unit norm)")
            else:
                print("  ⚠ Not normalized quaternions")

        # Sample movement analysis
        print("\n" + "-"*70)
        print("MOVEMENT ANALYSIS")
        print("-"*70)

        if reshaped.shape[0] > 1:
            # Calculate position changes between consecutive frames
            pos_diff = np.diff(positions, axis=0)
            movement = np.linalg.norm(pos_diff, axis=2)  # (frames-1, joints)

            avg_movement = movement.mean(axis=0)  # Average per joint

            print(f"\nAverage movement per joint (across all frames):")
            for joint_idx in range(min(10, num_joints)):
                print(f"  Joint {joint_idx:2d}: {avg_movement[joint_idx]:.6f} m/frame")

            most_active_joint = np.argmax(avg_movement)
            least_active_joint = np.argmin(avg_movement)

            print(f"\nMost active joint: {most_active_joint} "
                  f"({avg_movement[most_active_joint]:.6f} m/frame)")
            print(f"Least active joint: {least_active_joint} "
                  f"({avg_movement[least_active_joint]:.6f} m/frame)")

        # Save analysis results
        print("\n" + "-"*70)
        print("SAVING ANALYSIS")
        print("-"*70)

        output_dir = './test_conversion_output'

        # Save reshaped data
        reshaped_path = os.path.join(output_dir, 'sample_reshaped_25x14.npy')
        np.save(reshaped_path, reshaped)
        print(f"\n✓ Saved reshaped data: {reshaped_path}")
        print(f"  Shape: {reshaped.shape} (frames, 25 joints, 14 features)")

        # Save positions only
        positions_path = os.path.join(output_dir, 'sample_positions_25x3.npy')
        np.save(positions_path, positions)
        print(f"\n✓ Saved positions: {positions_path}")
        print(f"  Shape: {positions.shape} (frames, 25 joints, xyz)")

        # Create a summary dict
        summary = {
            'total_dims': 350,
            'num_joints': num_joints,
            'features_per_joint': features_per_joint,
            'num_frames': data.shape[0],
            'position_ranges': {
                'x': [float(positions[:, :, 0].min()), float(positions[:, :, 0].max())],
                'y': [float(positions[:, :, 1].min()), float(positions[:, :, 1].max())],
                'z': [float(positions[:, :, 2].min()), float(positions[:, :, 2].max())],
            }
        }

        import json
        summary_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Saved summary: {summary_path}")

        # Kinect v2 joint names for reference
        print("\n" + "-"*70)
        print("KINECT V2 JOINT MAPPING (for reference)")
        print("-"*70)

        kinect_v2_joints = [
            "SpineBase", "SpineMid", "Neck", "Head",
            "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
            "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
            "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
            "HipRight", "KneeRight", "AnkleRight", "FootRight",
            "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"
        ]

        print(f"\nIf this is Kinect v2, joints 0-24 are:")
        for i, joint_name in enumerate(kinect_v2_joints):
            print(f"  {i:2d}: {joint_name}")

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""

    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "SKELETON STREAM CONVERSION - TESTING SUITE" + " "*16 + "║")
    print("╚" + "="*68 + "╝")

    # Test 1: Load all streams
    file_stats = test_1_load_all_streams()

    # Test 2: Convert single file
    data = test_2_convert_single_file()

    # Test 3: Analyze structure
    if data is not None:
        test_3_analyze_kinect_structure(data)

    # Final summary
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

    if file_stats:
        print(f"✓ Test 1: Verified {len(file_stats)} stream files")
    else:
        print("✗ Test 1: Failed to load stream files")

    if data is not None:
        print(f"✓ Test 2: Converted sample file successfully")
        print(f"✓ Test 3: Analyzed structure (see output above)")
        print(f"\n✓ Output saved to: ./test_conversion_output/")
    else:
        print("✗ Test 2: Conversion failed")
        print("✗ Test 3: Skipped (no data)")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
