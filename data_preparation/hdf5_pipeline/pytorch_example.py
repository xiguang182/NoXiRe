"""
PyTorch training example showing how to use HDF5 data
with different feature types (OpenFace vs ViT).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Literal, Optional


class NoXiReHDF5Dataset(Dataset):
    """
    PyTorch Dataset for NoXiRe HDF5 data.
    Supports OpenFace, ViT, or mixed features.
    """

    def __init__(self,
                 hdf5_path: str,
                 person: Literal['expert', 'novice'] = 'expert',
                 feature_type: Literal['openface', 'vit', 'mixed'] = 'openface',
                 sequence_length: Optional[int] = None,
                 stride: int = 1):
        """
        Args:
            hdf5_path: Path to HDF5 file
            person: 'expert' or 'novice'
            feature_type: Which features to load
            sequence_length: Fixed sequence length (None = full sequence)
            stride: Stride for sliding window sampling
        """
        self.hdf5_path = hdf5_path
        self.person = person
        self.feature_type = feature_type
        self.sequence_length = sequence_length
        self.stride = stride

        # Load metadata
        with h5py.File(hdf5_path, 'r') as hf:
            self.num_samples = hf.attrs['num_samples']
            self.sample_names = [name.decode('utf-8') for name in hf['sample_names'][:]]

            # Get available features
            sample = hf['sample_000'][person]
            self.available_features = list(sample.keys())

        print(f"Dataset initialized:")
        print(f"  Samples: {self.num_samples}")
        print(f"  Person: {person}")
        print(f"  Features: {self.available_features}")
        print(f"  Sequence length: {sequence_length if sequence_length else 'Variable'}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Load a sample from HDF5."""
        with h5py.File(self.hdf5_path, 'r') as hf:
            sample = hf[f'sample_{idx:03d}'][self.person]

            if self.feature_type == 'openface':
                # Load OpenFace features
                face = torch.from_numpy(sample['face'][:])  # (T, 68, 2)
                aus = torch.from_numpy(sample['aus'][:])    # (T, 17)
                head = torch.from_numpy(sample['head'][:])  # (T, 6)

                # Concatenate into single feature vector
                face_flat = face.reshape(face.shape[0], -1)  # (T, 136)
                features = torch.cat([face_flat, aus, head], dim=-1)  # (T, 159)

            elif self.feature_type == 'vit':
                # Load ViT features (already in good shape)
                features = torch.from_numpy(sample['vit'][:])  # (T, D)

            elif self.feature_type == 'mixed':
                # Load both, return as dictionary
                face = torch.from_numpy(sample['face'][:])
                aus = torch.from_numpy(sample['aus'][:])
                head = torch.from_numpy(sample['head'][:])
                vit = torch.from_numpy(sample['vit'][:])

                face_flat = face.reshape(face.shape[0], -1)
                openface_features = torch.cat([face_flat, aus, head], dim=-1)

                features = {
                    'openface': openface_features,
                    'vit': vit
                }

            # Handle sequence length
            if self.sequence_length and not isinstance(features, dict):
                features = self._sample_sequence(features)
            elif self.sequence_length and isinstance(features, dict):
                features = {k: self._sample_sequence(v) for k, v in features.items()}

        return features, idx  # Return features and sample index

    def _sample_sequence(self, features):
        """Sample a fixed-length sequence from variable-length features."""
        T = features.shape[0]

        if T >= self.sequence_length:
            # Random crop
            start = np.random.randint(0, T - self.sequence_length + 1)
            return features[start:start + self.sequence_length]
        else:
            # Pad if sequence is too short
            padding = torch.zeros(self.sequence_length - T, *features.shape[1:])
            return torch.cat([features, padding], dim=0)


class OpenFaceModel(nn.Module):
    """Simple model for OpenFace features."""

    def __init__(self, input_dim: int = 159, hidden_dim: int = 256, num_classes: int = 7):
        super().__init__()
        # OpenFace features are in [0, 1] range (min-max scaled)
        # Can use directly with standard layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 159) - min-max scaled [0, 1]
        out, _ = self.lstm(x)
        # Take last timestep
        return self.fc(out[:, -1, :])


class ViTModel(nn.Module):
    """Simple model for ViT features."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 7):
        super().__init__()
        # ViT features keep original distribution (mean≈0, std≈1)
        # No additional normalization needed!
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 768) - original ViT distribution
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class MixedModel(nn.Module):
    """Two-stream model for OpenFace + ViT features."""

    def __init__(self,
                 openface_dim: int = 159,
                 vit_dim: int = 768,
                 hidden_dim: int = 256,
                 num_classes: int = 7):
        super().__init__()

        # Separate branches for different feature types
        # This allows each branch to handle its input distribution properly
        self.openface_branch = nn.Sequential(
            nn.Linear(openface_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.vit_branch = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temporal modeling
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, num_layers=2)

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, openface, vit):
        # openface: (B, T, 159) - [0, 1] range
        # vit: (B, T, 768) - original distribution

        B, T, _ = openface.shape

        # Process each timestep through branches
        openface_flat = openface.reshape(B * T, -1)
        vit_flat = vit.reshape(B * T, -1)

        openface_encoded = self.openface_branch(openface_flat)  # (B*T, 256)
        vit_encoded = self.vit_branch(vit_flat)                # (B*T, 256)

        # Concatenate
        combined = torch.cat([openface_encoded, vit_encoded], dim=-1)  # (B*T, 512)
        combined = combined.reshape(B, T, -1)  # (B, T, 512)

        # Temporal modeling
        out, _ = self.lstm(combined)

        # Classification
        return self.fusion(out[:, -1, :])


def train_openface_model():
    """Example: Training with OpenFace features."""
    print("="*60)
    print("Training with OpenFace features (min-max scaled)")
    print("="*60)

    # Dataset
    dataset = NoXiReHDF5Dataset(
        hdf5_path='./data/openface_only.h5',
        person='expert',
        feature_type='openface',
        sequence_length=100  # Fixed 100 frames
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Model
    model = OpenFaceModel(input_dim=159, hidden_dim=256, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for features, _ in dataloader:
            # features: (B, 100, 159) - min-max scaled [0, 1]

            # Dummy labels for example
            labels = torch.randint(0, 7, (features.shape[0],))

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def train_vit_model():
    """Example: Training with ViT features (no normalization)."""
    print("="*60)
    print("Training with ViT features (original distribution)")
    print("="*60)

    # Dataset
    dataset = NoXiReHDF5Dataset(
        hdf5_path='./data/vit_only.h5',
        person='expert',
        feature_type='vit',
        sequence_length=100
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Model
    model = ViTModel(input_dim=768, hidden_dim=256, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for features, _ in dataloader:
            # features: (B, 100, 768) - original ViT distribution (mean≈0)

            labels = torch.randint(0, 7, (features.shape[0],))

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def train_mixed_model():
    """Example: Training with both OpenFace and ViT features."""
    print("="*60)
    print("Training with mixed features (two-stream)")
    print("="*60)

    # Dataset
    dataset = NoXiReHDF5Dataset(
        hdf5_path='./data/mixed_features.h5',
        person='expert',
        feature_type='mixed',
        sequence_length=100
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Model
    model = MixedModel(openface_dim=159, vit_dim=768, hidden_dim=256, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for features, _ in dataloader:
            # features is a dict:
            # - 'openface': (B, 100, 159) - min-max scaled
            # - 'vit': (B, 100, 768) - original distribution

            labels = torch.randint(0, 7, (features['openface'].shape[0],))

            optimizer.zero_grad()
            outputs = model(features['openface'], features['vit'])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def inspect_batch_statistics():
    """Inspect actual statistics in training batches."""
    print("="*60)
    print("Inspecting batch statistics")
    print("="*60)

    # OpenFace
    print("\n1. OpenFace features:")
    dataset = NoXiReHDF5Dataset('./data/openface_only.h5', feature_type='openface')
    loader = DataLoader(dataset, batch_size=8)
    features, _ = next(iter(loader))
    print(f"   Shape: {features.shape}")
    print(f"   Min: {features.min():.4f} (should be ≈0)")
    print(f"   Max: {features.max():.4f} (should be ≈1)")
    print(f"   Mean: {features.mean():.4f}")
    print(f"   Std: {features.std():.4f}")

    # ViT
    print("\n2. ViT features:")
    dataset = NoXiReHDF5Dataset('./data/vit_only.h5', feature_type='vit')
    loader = DataLoader(dataset, batch_size=8)
    features, _ = next(iter(loader))
    print(f"   Shape: {features.shape}")
    print(f"   Min: {features.min():.4f}")
    print(f"   Max: {features.max():.4f}")
    print(f"   Mean: {features.mean():.4f} (should be ≈0)")
    print(f"   Std: {features.std():.4f} (should be ≈0.5-2)")


def compare_training_stability():
    """
    Compare training stability with different normalizations.
    This demonstrates why ViT features should be kept as-is.
    """
    print("="*60)
    print("Comparing training stability")
    print("="*60)

    # Simulate ViT features with different normalizations
    torch.manual_seed(42)

    # Original ViT distribution (mean≈0, std≈1)
    vit_original = torch.randn(16, 100, 768) * 1.0 + 0.0

    # After incorrect min-max scaling
    vit_minmax = (vit_original - vit_original.min()) / (vit_original.max() - vit_original.min())

    print("\nOriginal ViT distribution:")
    print(f"  Mean: {vit_original.mean():.4f}")
    print(f"  Std: {vit_original.std():.4f}")
    print(f"  Range: [{vit_original.min():.4f}, {vit_original.max():.4f}]")

    print("\nAfter min-max [0,1]:")
    print(f"  Mean: {vit_minmax.mean():.4f}")
    print(f"  Std: {vit_minmax.std():.4f}")
    print(f"  Range: [{vit_minmax.min():.4f}, {vit_minmax.max():.4f}]")

    # Train simple models
    model_original = nn.Linear(768, 7)
    model_minmax = nn.Linear(768, 7)

    criterion = nn.CrossEntropyLoss()
    optimizer_original = torch.optim.SGD(model_original.parameters(), lr=0.01)
    optimizer_minmax = torch.optim.SGD(model_minmax.parameters(), lr=0.01)

    labels = torch.randint(0, 7, (16,))

    print("\nTraining with original distribution:")
    for i in range(5):
        optimizer_original.zero_grad()
        out = model_original(vit_original[:, -1, :])
        loss = criterion(out, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_original.parameters(), float('inf'))
        optimizer_original.step()
        print(f"  Step {i+1}: Loss={loss.item():.4f}, GradNorm={grad_norm:.4f}")

    print("\nTraining with min-max [0,1]:")
    for i in range(5):
        optimizer_minmax.zero_grad()
        out = model_minmax(vit_minmax[:, -1, :])
        loss = criterion(out, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_minmax.parameters(), float('inf'))
        optimizer_minmax.step()
        print(f"  Step {i+1}: Loss={loss.item():.4f}, GradNorm={grad_norm:.4f}")

    print("\n→ Notice: Gradient norms and training dynamics differ!")
    print("→ Original distribution is what the model expects.")


if __name__ == '__main__':
    print("PyTorch Training Examples\n")

    # Uncomment to run:
    # train_openface_model()
    # train_vit_model()
    # train_mixed_model()
    # inspect_batch_statistics()

    compare_training_stability()
