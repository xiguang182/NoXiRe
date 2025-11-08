import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Literal


def min_max_scaling_along_axis(arr, axis):
    """Min-max scaling for handcrafted features like OpenFace."""
    min_vals = np.min(arr, axis=axis, keepdims=True)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    scaled_data = (arr - min_vals) / (max_vals - min_vals + 1e-7)
    return np.round(scaled_data, decimals=4)


def standardize_features(arr, axis=0, store_stats=False):
    """
    Z-score standardization: (x - mean) / std
    Use this for features that need centering (optional for ViT).

    Args:
        arr: Input array
        axis: Axis along which to compute statistics
        store_stats: If True, return (scaled_data, mean, std) for later denormalization
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    scaled_data = (arr - mean) / (std + 1e-7)

    if store_stats:
        return scaled_data, mean.squeeze(), std.squeeze()
    return scaled_data


def format_openface_from_df(df):
    """Extract and normalize OpenFace features (handcrafted features)."""
    # 68 x,y pairs face points - min-max scaling
    face = df.iloc[:, 299:435].values
    face = face.reshape(-1, 2, 68)
    scaled_face = min_max_scaling_along_axis(arr=face, axis=2)
    scaled_face = scaled_face.transpose(0, 2, 1)

    # 17 AU intensity, since the scale is 0-5, rescale to 0-1
    aus = df.iloc[:, 679:696].values / 5

    # head pose coordinate - min-max scaling
    head = min_max_scaling_along_axis(arr=df.iloc[:, 293:296].values, axis=0)
    direction = min_max_scaling_along_axis(arr=df.iloc[:, 296:299].values, axis=1)
    head = np.concatenate((head, direction), axis=1)

    return {'face': scaled_face, 'aus': aus, 'head': head}


def format_vit_features(features: np.ndarray,
                        normalization: Literal['none', 'standardize', 'minmax'] = 'none'):
    """
    Format ViT latent features with appropriate normalization.

    Args:
        features: Raw ViT features (T, D) where T=frames, D=embedding dimension
        normalization:
            - 'none': Keep as-is (RECOMMENDED for ViT)
            - 'standardize': Z-score normalization (use if needed for compatibility)
            - 'minmax': Min-max scaling (NOT recommended for ViT)

    Returns:
        Formatted features
    """
    if normalization == 'none':
        return features
    elif normalization == 'standardize':
        return standardize_features(features, axis=0)
    elif normalization == 'minmax':
        return min_max_scaling_along_axis(features, axis=0)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def save_to_hdf5(
    output_path: str = './data/openface_data.h5',
    feature_type: Literal['openface', 'vit', 'mixed'] = 'openface',
    vit_normalization: Literal['none', 'standardize', 'minmax'] = 'none',
    vit_features_path: Optional[str] = None
):
    """
    Save data to HDF5 format with flexible feature handling.

    Args:
        output_path: Path to save HDF5 file
        feature_type:
            - 'openface': Only OpenFace features (original behavior)
            - 'vit': Only ViT features
            - 'mixed': Both OpenFace and ViT features
        vit_normalization: How to normalize ViT features (recommend 'none')
        vit_features_path: Path to directory containing ViT features (.npy files)
    """
    data_folder = './data/openface'
    csv_list = './data/sample_list.csv'
    sample_list = pd.read_csv(csv_list).iloc[:, 1].values

    with h5py.File(output_path, 'w') as hf:
        # Store metadata
        hf.attrs['num_samples'] = len(sample_list)
        hf.attrs['feature_type'] = feature_type
        hf.attrs['vit_normalization'] = vit_normalization

        if feature_type == 'openface':
            hf.attrs['description'] = 'NoXiRe OpenFace features (min-max normalized)'
            hf.attrs['features'] = 'face: (T, 68, 2), aus: (T, 17), head: (T, 6)'
        elif feature_type == 'vit':
            hf.attrs['description'] = f'NoXiRe ViT latent features (normalization: {vit_normalization})'
            hf.attrs['features'] = 'vit: (T, D) where D is embedding dimension'
        else:  # mixed
            hf.attrs['description'] = 'NoXiRe OpenFace + ViT features'
            hf.attrs['features'] = 'OpenFace: face/aus/head, ViT: vit'

        # Create sample names dataset
        sample_names_dataset = hf.create_dataset(
            'sample_names',
            data=[s.encode('utf-8') for s in sample_list],
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        for idx, item in enumerate(tqdm(sample_list, desc='Saving to HDF5', leave=True)):
            sample_group = hf.create_group(f'sample_{idx:03d}')
            sample_group.attrs['sample_name'] = item

            # Process expert and novice
            for person in ['expert', 'novice']:
                person_group = sample_group.create_group(person)

                # OpenFace features
                if feature_type in ['openface', 'mixed']:
                    csv_file = os.path.join(data_folder, f'{item}_{person}.csv')
                    df = pd.read_csv(csv_file)
                    openface_data = format_openface_from_df(df)

                    person_group.create_dataset(
                        'face',
                        data=openface_data['face'],
                        compression='gzip',
                        compression_opts=4,
                        dtype='float32'
                    )
                    person_group.create_dataset(
                        'aus',
                        data=openface_data['aus'],
                        compression='gzip',
                        compression_opts=4,
                        dtype='float32'
                    )
                    person_group.create_dataset(
                        'head',
                        data=openface_data['head'],
                        compression='gzip',
                        compression_opts=4,
                        dtype='float32'
                    )

                # ViT features
                if feature_type in ['vit', 'mixed'] and vit_features_path:
                    vit_file = os.path.join(vit_features_path, f'{item}_{person}.npy')
                    if os.path.exists(vit_file):
                        vit_raw = np.load(vit_file)
                        vit_formatted = format_vit_features(vit_raw, vit_normalization)

                        person_group.create_dataset(
                            'vit',
                            data=vit_formatted,
                            compression='gzip',
                            compression_opts=4,
                            dtype='float32'
                        )
                        person_group['vit'].attrs['normalization'] = vit_normalization
                        person_group['vit'].attrs['embedding_dim'] = vit_formatted.shape[-1]

                # Store frame count
                if 'face' in person_group:
                    person_group.attrs['num_frames'] = person_group['face'].shape[0]
                elif 'vit' in person_group:
                    person_group.attrs['num_frames'] = person_group['vit'].shape[0]

    print(f"Data saved to {output_path}")
    print(f"Feature type: {feature_type}")
    print(f"ViT normalization: {vit_normalization}")
    return output_path


def load_sample(hdf5_path: str,
                sample_idx: int,
                person: str = 'expert',
                features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load a specific sample from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        sample_idx: Sample index (0-based)
        person: 'expert' or 'novice'
        features: List of features to load (None = all available)
                 Options: ['face', 'aus', 'head', 'vit']

    Returns:
        Dictionary with requested features
    """
    with h5py.File(hdf5_path, 'r') as hf:
        sample_group = hf[f'sample_{sample_idx:03d}']
        person_group = sample_group[person]

        # Determine which features to load
        available_features = list(person_group.keys())
        if features is None:
            features_to_load = available_features
        else:
            features_to_load = [f for f in features if f in available_features]

        # Load requested features
        data = {}
        for feature in features_to_load:
            data[feature] = person_group[feature][:]

            # Add metadata for ViT features
            if feature == 'vit' and 'normalization' in person_group[feature].attrs:
                data['vit_normalization'] = person_group[feature].attrs['normalization']

    return data


def get_feature_info(hdf5_path: str):
    """Get information about features stored in HDF5 file."""
    with h5py.File(hdf5_path, 'r') as hf:
        print(f"Dataset: {hdf5_path}")
        print(f"Feature type: {hf.attrs.get('feature_type', 'unknown')}")
        print(f"Description: {hf.attrs.get('description', 'N/A')}")
        print(f"Number of samples: {hf.attrs.get('num_samples', 'N/A')}")

        if 'sample_000' in hf:
            print("\nAvailable features:")
            sample = hf['sample_000']
            for person in ['expert', 'novice']:
                if person in sample:
                    print(f"\n  {person.capitalize()}:")
                    for feature in sample[person].keys():
                        dataset = sample[person][feature]
                        print(f"    {feature}: shape={dataset.shape}, dtype={dataset.dtype}")

                        # Show ViT-specific info
                        if feature == 'vit' and 'normalization' in dataset.attrs:
                            print(f"      normalization={dataset.attrs['normalization']}")
                            print(f"      embedding_dim={dataset.attrs.get('embedding_dim', 'N/A')}")


def main():
    """Example usage for different feature types."""

    print("="*60)
    print("EXAMPLE 1: OpenFace features only (original behavior)")
    print("="*60)
    openface_path = save_to_hdf5(
        output_path='./data/openface_only.h5',
        feature_type='openface'
    )
    print("\n")
    get_feature_info(openface_path)

    print("\n\n" + "="*60)
    print("EXAMPLE 2: ViT features with no normalization (RECOMMENDED)")
    print("="*60)
    print("If you have ViT features as .npy files in './data/vit_features/':")
    print("  vit_path = save_to_hdf5(")
    print("      output_path='./data/vit_only.h5',")
    print("      feature_type='vit',")
    print("      vit_normalization='none',  # Keep as-is")
    print("      vit_features_path='./data/vit_features/'")
    print("  )")

    print("\n\n" + "="*60)
    print("EXAMPLE 3: Mixed features (OpenFace + ViT)")
    print("="*60)
    print("  mixed_path = save_to_hdf5(")
    print("      output_path='./data/mixed_features.h5',")
    print("      feature_type='mixed',")
    print("      vit_normalization='none',")
    print("      vit_features_path='./data/vit_features/'")
    print("  )")
    print("\n")
    print("Then load with:")
    print("  # Load only ViT features")
    print("  data = load_sample('./data/mixed_features.h5', 0, 'expert', features=['vit'])")
    print("  # Load OpenFace + ViT")
    print("  data = load_sample('./data/mixed_features.h5', 0, 'expert')")


if __name__ == '__main__':
    main()
