"""
HDF5 Data Pipeline for NoXiRe Dataset

This package provides efficient HDF5-based data storage with support for:
- OpenFace features (with min-max normalization)
- ViT latent features (no normalization - use as-is)
- Mixed features (different normalization per type)

Quick Start:
    from hdf5_pipeline.data_hdf5_flexible import save_to_hdf5, load_sample

    # Save ViT features (no normalization)
    save_to_hdf5(
        output_path='./data/vit.h5',
        feature_type='vit',
        vit_normalization='none',  # Use as-is!
        vit_features_path='./data/vit_features/'
    )

    # Load sample
    data = load_sample('./data/vit.h5', sample_idx=0, person='expert')

Documentation:
    See README.md in this folder for complete guide.
"""

__version__ = '1.0.0'
__author__ = 'NoXiRe Team'

# Import main functions for convenience
try:
    from .data_hdf5_flexible import (
        save_to_hdf5,
        load_sample,
        load_sample_by_name,
        load_slice,
        get_feature_info,
        search_samples_by_frame_count
    )
except ImportError:
    # If dependencies not installed
    pass
