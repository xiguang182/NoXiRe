import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

def min_max_scaling_along_axis(arr, axis):
    min_vals = np.min(arr, axis=axis, keepdims=True)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    scaled_data = (arr - min_vals) / (max_vals - min_vals + 1e-7)
    return np.round(scaled_data, decimals=4)

def format_from_df(df):
    # 68 x,y pairs face points
    face = df.iloc[:, 299:435].values
    face = face.reshape(-1, 2, 68)
    scaled_face = min_max_scaling_along_axis(arr=face, axis=2)
    scaled_face = scaled_face.transpose(0, 2, 1)

    # 17 AU intensity, since the scale is 0-5, rescale to 0-1
    aus = df.iloc[:, 679:696].values / 5

    # head pose coordinate
    head = min_max_scaling_along_axis(arr=df.iloc[:, 293:296].values, axis=0)
    direction = min_max_scaling_along_axis(arr=df.iloc[:, 296:299].values, axis=1)
    head = np.concatenate((head, direction), axis=1)
    return {'face': scaled_face, 'aus': aus, 'head': head}


def save_to_hdf5(output_path='./data/openface_data.h5'):
    """
    Save data to HDF5 format with the following structure:

    /sample_000/
        /expert/
            face: (T, 68, 2) - facial landmarks
            aus: (T, 17) - action units
            head: (T, 6) - head pose
        /novice/
            face: (T, 68, 2)
            aus: (T, 17)
            head: (T, 6)
    /sample_001/
        ...

    Where T is the number of frames (varies per sample)
    """
    data_folder = './data/openface'
    csv_list = './data/sample_list.csv'
    sample_list = pd.read_csv(csv_list).iloc[:, 1].values

    with h5py.File(output_path, 'w') as hf:
        # Store metadata
        hf.attrs['num_samples'] = len(sample_list)
        hf.attrs['description'] = 'NoXiRe OpenFace data with expert and novice facial features'
        hf.attrs['features'] = 'face: (T, 68, 2), aus: (T, 17), head: (T, 6)'

        # Create a dataset to store sample names for easy lookup
        sample_names_dataset = hf.create_dataset(
            'sample_names',
            data=[s.encode('utf-8') for s in sample_list],
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        for idx, item in enumerate(tqdm(sample_list, desc='Saving to HDF5', leave=True)):
            # Read CSV files
            tmp_exp = pd.read_csv(os.path.join(data_folder, item + '_expert.csv'))
            tmp_nov = pd.read_csv(os.path.join(data_folder, item + '_novice.csv'))

            # Format data
            exp_data = format_from_df(tmp_exp)
            nov_data = format_from_df(tmp_nov)

            # Create group for this sample
            sample_group = hf.create_group(f'sample_{idx:03d}')
            sample_group.attrs['sample_name'] = item
            sample_group.attrs['num_frames_expert'] = exp_data['face'].shape[0]
            sample_group.attrs['num_frames_novice'] = nov_data['face'].shape[0]

            # Save expert data
            exp_group = sample_group.create_group('expert')
            exp_group.create_dataset(
                'face',
                data=exp_data['face'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )
            exp_group.create_dataset(
                'aus',
                data=exp_data['aus'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )
            exp_group.create_dataset(
                'head',
                data=exp_data['head'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )

            # Save novice data
            nov_group = sample_group.create_group('novice')
            nov_group.create_dataset(
                'face',
                data=nov_data['face'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )
            nov_group.create_dataset(
                'aus',
                data=nov_data['aus'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )
            nov_group.create_dataset(
                'head',
                data=nov_data['head'],
                compression='gzip',
                compression_opts=4,
                dtype='float32'
            )

    print(f"Data saved to {output_path}")
    return output_path


def load_sample(hdf5_path, sample_idx, person='expert'):
    """
    Load a specific sample from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        sample_idx: Sample index (0-based)
        person: 'expert' or 'novice'

    Returns:
        Dictionary with 'face', 'aus', 'head' arrays
    """
    with h5py.File(hdf5_path, 'r') as hf:
        sample_group = hf[f'sample_{sample_idx:03d}']
        person_group = sample_group[person]

        data = {
            'face': person_group['face'][:],
            'aus': person_group['aus'][:],
            'head': person_group['head'][:]
        }
    return data


def load_sample_by_name(hdf5_path, sample_name, person='expert'):
    """
    Load a specific sample by name from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        sample_name: Name of the sample (e.g., from sample_list.csv)
        person: 'expert' or 'novice'

    Returns:
        Dictionary with 'face', 'aus', 'head' arrays
    """
    with h5py.File(hdf5_path, 'r') as hf:
        sample_names = [name.decode('utf-8') for name in hf['sample_names'][:]]
        if sample_name not in sample_names:
            raise ValueError(f"Sample '{sample_name}' not found in dataset")

        sample_idx = sample_names.index(sample_name)
        sample_group = hf[f'sample_{sample_idx:03d}']
        person_group = sample_group[person]

        data = {
            'face': person_group['face'][:],
            'aus': person_group['aus'][:],
            'head': person_group['head'][:]
        }
    return data


def load_slice(hdf5_path, sample_idx, person='expert', feature='face',
               frame_start=None, frame_end=None):
    """
    Load a slice of data without loading the entire dataset.

    Args:
        hdf5_path: Path to HDF5 file
        sample_idx: Sample index (0-based)
        person: 'expert' or 'novice'
        feature: 'face', 'aus', or 'head'
        frame_start: Starting frame (None for beginning)
        frame_end: Ending frame (None for end)

    Returns:
        Sliced array
    """
    with h5py.File(hdf5_path, 'r') as hf:
        sample_group = hf[f'sample_{sample_idx:03d}']
        person_group = sample_group[person]

        if frame_start is None and frame_end is None:
            return person_group[feature][:]
        elif frame_end is None:
            return person_group[feature][frame_start:]
        elif frame_start is None:
            return person_group[feature][:frame_end]
        else:
            return person_group[feature][frame_start:frame_end]


def get_dataset_info(hdf5_path):
    """
    Print information about the HDF5 dataset.

    Args:
        hdf5_path: Path to HDF5 file
    """
    with h5py.File(hdf5_path, 'r') as hf:
        print(f"Dataset: {hdf5_path}")
        print(f"Description: {hf.attrs.get('description', 'N/A')}")
        print(f"Number of samples: {hf.attrs.get('num_samples', 'N/A')}")
        print(f"Features: {hf.attrs.get('features', 'N/A')}")
        print("\nSample structure:")

        # Show first sample structure
        if 'sample_000' in hf:
            sample = hf['sample_000']
            print(f"  Sample name: {sample.attrs.get('sample_name', 'N/A')}")
            print(f"  Expert frames: {sample.attrs.get('num_frames_expert', 'N/A')}")
            print(f"  Novice frames: {sample.attrs.get('num_frames_novice', 'N/A')}")

            for person in ['expert', 'novice']:
                if person in sample:
                    print(f"\n  {person.capitalize()}:")
                    for feature in sample[person].keys():
                        dataset = sample[person][feature]
                        print(f"    {feature}: shape={dataset.shape}, dtype={dataset.dtype}")


def search_samples_by_frame_count(hdf5_path, person='expert',
                                   min_frames=None, max_frames=None):
    """
    Search for samples with specific frame count ranges.

    Args:
        hdf5_path: Path to HDF5 file
        person: 'expert' or 'novice'
        min_frames: Minimum number of frames (None for no minimum)
        max_frames: Maximum number of frames (None for no maximum)

    Returns:
        List of (sample_idx, sample_name, num_frames) tuples
    """
    results = []
    with h5py.File(hdf5_path, 'r') as hf:
        num_samples = hf.attrs['num_samples']

        for idx in range(num_samples):
            sample_group = hf[f'sample_{idx:03d}']
            sample_name = sample_group.attrs['sample_name']

            if person == 'expert':
                num_frames = sample_group.attrs['num_frames_expert']
            else:
                num_frames = sample_group.attrs['num_frames_novice']

            if min_frames is not None and num_frames < min_frames:
                continue
            if max_frames is not None and num_frames > max_frames:
                continue

            results.append((idx, sample_name, num_frames))

    return results


def main():
    """Main function to convert pickle data to HDF5."""
    output_path = save_to_hdf5('./data/openface_data.h5')

    print("\n" + "="*60)
    get_dataset_info(output_path)

    print("\n" + "="*60)
    print("Example usage:")
    print("\n1. Load specific sample (index-based):")
    data = load_sample(output_path, sample_idx=0, person='expert')
    print(f"   Face shape: {data['face'].shape}")
    print(f"   AUs shape: {data['aus'].shape}")
    print(f"   Head shape: {data['head'].shape}")

    print("\n2. Load slice of frames (frames 10-20 of face data):")
    face_slice = load_slice(output_path, sample_idx=0, person='expert',
                           feature='face', frame_start=10, frame_end=20)
    print(f"   Face slice shape: {face_slice.shape}")

    print("\n3. Search samples by frame count:")
    samples = search_samples_by_frame_count(output_path, person='expert',
                                           min_frames=100, max_frames=500)
    print(f"   Found {len(samples)} samples with 100-500 frames")
    if samples:
        print(f"   First result: idx={samples[0][0]}, name={samples[0][1]}, frames={samples[0][2]}")


if __name__ == '__main__':
    main()
