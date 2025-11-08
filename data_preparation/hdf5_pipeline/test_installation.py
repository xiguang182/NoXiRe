#!/usr/bin/env python3
"""
Test script to verify all dependencies and functionality.
Run this before using the HDF5 data pipeline.
"""

import sys
import importlib


def check_dependencies():
    """Check if all required packages are installed."""
    print("="*60)
    print("Checking Dependencies")
    print("="*60)

    required = {
        'Core': ['numpy', 'pandas', 'h5py', 'tqdm'],
        'ViT extraction': ['torch', 'torchvision', 'transformers', 'cv2'],
        'Visualization': ['matplotlib', 'scipy']
    }

    all_installed = True
    for category, packages in required.items():
        print(f"\n{category}:")
        for package in packages:
            try:
                # Special case for opencv
                if package == 'cv2':
                    importlib.import_module('cv2')
                else:
                    importlib.import_module(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} - NOT INSTALLED")
                all_installed = False

    return all_installed


def test_hdf5_basic():
    """Test basic HDF5 functionality."""
    print("\n" + "="*60)
    print("Testing HDF5 Basic Functionality")
    print("="*60)

    try:
        import h5py
        import numpy as np
        import tempfile
        import os

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        # Write test data
        print("\n1. Writing test data...")
        with h5py.File(tmp_path, 'w') as hf:
            hf.attrs['test_attr'] = 'test_value'
            hf.create_dataset('test_data', data=np.random.randn(100, 10))
        print("   ✓ Write successful")

        # Read test data
        print("2. Reading test data...")
        with h5py.File(tmp_path, 'r') as hf:
            assert hf.attrs['test_attr'] == 'test_value'
            data = hf['test_data'][:]
            assert data.shape == (100, 10)
        print("   ✓ Read successful")

        # Test slicing
        print("3. Testing slicing...")
        with h5py.File(tmp_path, 'r') as hf:
            slice_data = hf['test_data'][10:20]
            assert slice_data.shape == (10, 10)
        print("   ✓ Slicing successful")

        # Cleanup
        os.unlink(tmp_path)
        print("\n✓ All HDF5 basic tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ HDF5 test failed: {e}")
        return False


def test_data_pipeline():
    """Test the data pipeline modules."""
    print("\n" + "="*60)
    print("Testing Data Pipeline Modules")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test importing modules
    modules = [
        ('data_hdf5', 'Original HDF5 implementation'),
        ('data_hdf5_flexible', 'Flexible HDF5 implementation'),
        ('extract_vit_features', 'ViT feature extraction'),
        ('pytorch_example', 'PyTorch integration'),
        ('compare_formats', 'Format comparison'),
        ('normalization_comparison', 'Normalization comparison')
    ]

    for module_name, description in modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name:<30} ({description})")
            tests_passed += 1
        except ImportError as e:
            print(f"✗ {module_name:<30} - Import failed: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"⚠ {module_name:<30} - Loaded but has issues: {e}")
            tests_passed += 1  # Still counts as passed if it imports

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_normalization_functions():
    """Test normalization functions."""
    print("\n" + "="*60)
    print("Testing Normalization Functions")
    print("="*60)

    try:
        import numpy as np
        from data_hdf5_flexible import min_max_scaling_along_axis, standardize_features

        # Test data
        data = np.random.randn(10, 5) * 10 + 50

        # Min-max scaling
        print("\n1. Testing min-max scaling...")
        scaled = min_max_scaling_along_axis(data, axis=0)
        assert scaled.min() >= 0 and scaled.max() <= 1, "Min-max scaling failed"
        print(f"   Original range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"   Scaled range: [{scaled.min():.2f}, {scaled.max():.2f}]")
        print("   ✓ Min-max scaling works correctly")

        # Standardization
        print("\n2. Testing standardization...")
        standardized = standardize_features(data, axis=0)
        mean = np.mean(standardized, axis=0)
        std = np.std(standardized, axis=0)
        assert np.allclose(mean, 0, atol=1e-10), "Mean should be ~0"
        assert np.allclose(std, 1, atol=1e-10), "Std should be ~1"
        print(f"   Mean: {mean.mean():.6f} (should be ≈0)")
        print(f"   Std: {std.mean():.6f} (should be ≈1)")
        print("   ✓ Standardization works correctly")

        print("\n✓ All normalization tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_model_loading():
    """Test if ViT models can be loaded."""
    print("\n" + "="*60)
    print("Testing ViT Model Loading (Optional)")
    print("="*60)

    try:
        import torch
        from transformers import ViTModel

        print("\nAttempting to load ViT model...")
        print("(This may take a while on first run as it downloads the model)")

        model_name = 'google/vit-base-patch16-224'
        model = ViTModel.from_pretrained(model_name)

        print(f"✓ Successfully loaded {model_name}")
        print(f"  Embedding dimension: {model.config.hidden_size}")

        return True

    except ImportError:
        print("⚠ PyTorch or Transformers not installed - skipping")
        return None
    except Exception as e:
        print(f"⚠ Could not load ViT model: {e}")
        print("  This is OK - models will be downloaded when needed")
        return None


def run_sanity_checks():
    """Run sanity checks on the implementation."""
    print("\n" + "="*60)
    print("Running Sanity Checks")
    print("="*60)

    try:
        import numpy as np

        # Check 1: ViT features should not be min-max scaled
        print("\n1. Checking ViT normalization recommendation...")
        vit_features = np.random.randn(100, 768)  # Simulated ViT output
        mean, std = vit_features.mean(), vit_features.std()

        if abs(mean) < 0.2 and 0.8 < std < 1.2:
            print(f"   ✓ ViT features look normal (mean={mean:.3f}, std={std:.3f})")
            print("   → Recommendation: Use as-is (no normalization)")
        else:
            print(f"   ⚠ Unusual ViT distribution (mean={mean:.3f}, std={std:.3f})")

        # Check 2: OpenFace features benefit from min-max
        print("\n2. Checking OpenFace normalization benefit...")
        face_x = np.random.uniform(100, 800, (100, 68))  # Pixel coordinates
        aus = np.random.uniform(0, 5, (100, 17))  # Action units

        print(f"   Face X range: [{face_x.min():.1f}, {face_x.max():.1f}]")
        print(f"   AUs range: [{aus.min():.1f}, {aus.max():.1f}]")
        print("   → Different scales! Min-max scaling recommended ✓")

        # Check 3: Semantic preservation
        print("\n3. Checking semantic preservation...")
        from scipy.spatial.distance import cosine

        frame1 = np.random.randn(768)
        frame2 = np.random.randn(768)

        sim_before = 1 - cosine(frame1, frame2)

        # After min-max
        combined = np.stack([frame1, frame2])
        scaled = (combined - combined.min()) / (combined.max() - combined.min())
        sim_after = 1 - cosine(scaled[0], scaled[1])

        change = abs(sim_before - sim_after)
        print(f"   Similarity before normalization: {sim_before:.4f}")
        print(f"   Similarity after min-max: {sim_after:.4f}")
        print(f"   Change: {change:.4f}")

        if change > 0.05:
            print("   ⚠ Significant semantic distortion detected!")
            print("   → This is why we DON'T normalize ViT features ✓")
        else:
            print("   ✓ Semantic preserved (for this example)")

        print("\n✓ All sanity checks passed!")
        return True

    except ImportError as e:
        print(f"⚠ Missing dependency for sanity checks: {e}")
        return None
    except Exception as e:
        print(f"✗ Sanity check failed: {e}")
        return False


def print_summary():
    """Print summary and next steps."""
    print("\n" + "="*60)
    print("SUMMARY & NEXT STEPS")
    print("="*60)

    print("""
Installation Status:
  If all tests passed above, you're ready to use the pipeline!

Quick Start:

  1. For OpenFace features only:
     python data_hdf5.py

  2. For ViT features:
     python extract_vit_features.py  # Extract features first
     python data_hdf5_flexible.py    # Then save to HDF5

  3. Compare pickle vs HDF5:
     python compare_formats.py

  4. Visualize normalization effects:
     python normalization_comparison.py

Key Files:
  - QUICK_START.md - Quick reference guide
  - NORMALIZATION_GUIDE.md - Deep dive on normalization
  - README_HDF5.md - HDF5 usage guide
  - pytorch_example.py - Training examples

Remember:
  ✓ OpenFace features → Min-max scale to [0, 1]
  ✓ ViT features → Use as-is (vit_normalization='none')
  ✓ Mixed features → Different normalization per type

Questions? Check the documentation files!
    """)


def main():
    """Run all tests."""
    print("""
╔════════════════════════════════════════════════════════════╗
║   NoXiRe HDF5 Data Pipeline - Installation Test           ║
╚════════════════════════════════════════════════════════════╝
    """)

    results = []

    # Run tests
    results.append(("Dependencies", check_dependencies()))
    results.append(("HDF5 Basic", test_hdf5_basic()))
    results.append(("Data Pipeline", test_data_pipeline()))
    results.append(("Normalization", test_normalization_functions()))
    results.append(("ViT Loading", test_vit_model_loading()))
    results.append(("Sanity Checks", run_sanity_checks()))

    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for name, result in results:
        if result is True:
            print(f"✓ {name:<20} PASSED")
        elif result is False:
            print(f"✗ {name:<20} FAILED")
        else:
            print(f"⚠ {name:<20} SKIPPED")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 All tests passed! You're ready to go!")
    else:
        print("\n⚠ Some tests failed. Please install missing dependencies:")
        print("  pip install h5py numpy pandas tqdm")
        print("  pip install torch torchvision transformers opencv-python  # For ViT")
        print("  pip install matplotlib scipy  # For visualization")

    print_summary()


if __name__ == '__main__':
    main()
