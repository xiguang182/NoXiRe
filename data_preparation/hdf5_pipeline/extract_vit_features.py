"""
Example script to extract ViT features from video frames.
This demonstrates the recommended approach for extracting and storing ViT embeddings.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import os


class ViTFeatureExtractor:
    """Extract features from pre-trained ViT models."""

    def __init__(self,
                 model_name: str = 'google/vit-base-patch16-224',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ViT feature extractor.

        Args:
            model_name: HuggingFace model name
                - 'google/vit-base-patch16-224' (768-dim, 86M params)
                - 'google/vit-large-patch16-224' (1024-dim, 304M params)
                - 'facebook/dino-vitb16' (768-dim, self-supervised)
                - 'openai/clip-vit-base-patch32' (512-dim, multimodal)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading ViT model: {model_name}")
        self.model = ViTModel.from_pretrained(model_name).to(device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)

        # Set to evaluation mode
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_output = self.model(dummy_input).last_hidden_state
            self.embedding_dim = dummy_output.shape[-1]

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {device}")

    @torch.no_grad()
    def extract_from_frames(self, frames: np.ndarray, use_cls_token: bool = True) -> np.ndarray:
        """
        Extract ViT features from video frames.

        Args:
            frames: Array of frames (T, H, W, 3) or list of frames
            use_cls_token: If True, use [CLS] token (recommended)
                          If False, use mean of all patch tokens

        Returns:
            features: (T, embedding_dim) array
        """
        features = []

        for frame in tqdm(frames, desc='Extracting features', leave=False):
            # Preprocess frame
            inputs = self.processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Extract features
            outputs = self.model(**inputs)

            if use_cls_token:
                # Use [CLS] token (first token) - most common approach
                feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                # Use mean of all patch tokens
                feature = outputs.last_hidden_state[:, 1:, :].mean(dim=1).cpu().numpy()

            features.append(feature.squeeze())

        return np.array(features)

    @torch.no_grad()
    def extract_from_video(self, video_path: str, use_cls_token: bool = True) -> np.ndarray:
        """
        Extract features from a video file.

        Args:
            video_path: Path to video file
            use_cls_token: Use [CLS] token or mean pooling

        Returns:
            features: (T, embedding_dim) array
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in {video_path}")

        return self.extract_from_frames(frames, use_cls_token)


def extract_all_samples(
    video_folder: str,
    sample_list_csv: str,
    output_folder: str,
    model_name: str = 'google/vit-base-patch16-224'
):
    """
    Extract ViT features for all samples in the dataset.

    Args:
        video_folder: Folder containing video files
        sample_list_csv: CSV file with sample names
        output_folder: Folder to save .npy feature files
        model_name: ViT model to use
    """
    import pandas as pd

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load sample list
    sample_list = pd.read_csv(sample_list_csv).iloc[:, 1].values

    # Initialize extractor
    extractor = ViTFeatureExtractor(model_name=model_name)

    # Extract features for each sample
    for item in tqdm(sample_list, desc='Processing samples'):
        for person in ['expert', 'novice']:
            video_path = os.path.join(video_folder, f'{item}_{person}.mp4')
            output_path = os.path.join(output_folder, f'{item}_{person}.npy')

            if os.path.exists(output_path):
                print(f"Skipping {item}_{person} (already exists)")
                continue

            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            try:
                # Extract features
                features = extractor.extract_from_video(video_path)

                # Save as numpy array (no normalization!)
                np.save(output_path, features)

                print(f"Saved {item}_{person}: shape={features.shape}")

            except Exception as e:
                print(f"Error processing {video_path}: {e}")


def inspect_vit_features(npy_path: str):
    """
    Inspect statistics of extracted ViT features.
    Use this to verify features are properly extracted.
    """
    features = np.load(npy_path)

    print(f"\nFeature file: {npy_path}")
    print(f"Shape: {features.shape} (T={features.shape[0]}, D={features.shape[1]})")
    print(f"\nStatistics:")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std:  {features.std():.4f}")
    print(f"  Min:  {features.min():.4f}")
    print(f"  Max:  {features.max():.4f}")

    # Per-dimension statistics
    mean_per_dim = features.mean(axis=0)
    std_per_dim = features.std(axis=0)

    print(f"\nPer-dimension statistics:")
    print(f"  Mean of means: {mean_per_dim.mean():.4f}")
    print(f"  Mean of stds:  {std_per_dim.mean():.4f}")

    # Check if normalized
    print(f"\nNormalization check:")
    if abs(features.mean()) < 0.1 and 0.5 < features.std() < 2.0:
        print("  ✓ Features appear to be from a normalized model (LayerNorm)")
        print("  → Recommended: Use as-is (no additional normalization)")
    else:
        print("  ⚠ Features have unusual statistics")
        print("  → Double-check your extraction code")

    # Temporal consistency check
    if features.shape[0] > 1:
        from scipy.spatial.distance import cosine
        similarities = []
        for i in range(min(10, features.shape[0] - 1)):
            sim = 1 - cosine(features[i], features[i+1])
            similarities.append(sim)

        print(f"\nTemporal consistency (adjacent frame similarity):")
        print(f"  Mean: {np.mean(similarities):.4f}")
        print(f"  Std:  {np.std(similarities):.4f}")
        if np.mean(similarities) > 0.9:
            print("  ✓ High temporal consistency (expected for video)")
        else:
            print("  ⚠ Low temporal consistency (check video sampling)")


def example_usage():
    """Example of how to use the feature extractor."""

    print("="*60)
    print("EXAMPLE 1: Extract from a single video")
    print("="*60)

    # Initialize extractor
    extractor = ViTFeatureExtractor(model_name='google/vit-base-patch16-224')

    # Option A: From video file
    video_path = './data/videos/001.001.001.001_expert.mp4'
    if os.path.exists(video_path):
        features = extractor.extract_from_video(video_path)
        print(f"Extracted features: {features.shape}")

        # Save (no normalization!)
        np.save('./data/vit_features/001.001.001.001_expert.npy', features)
        print("Features saved!")

    # Option B: From frame array
    # Assume you have frames from OpenFace CSV or other source
    # frames = [...] # (T, H, W, 3)
    # features = extractor.extract_from_frames(frames)

    print("\n" + "="*60)
    print("EXAMPLE 2: Extract all samples in dataset")
    print("="*60)

    # Extract all
    extract_all_samples(
        video_folder='./data/videos/',
        sample_list_csv='./data/sample_list.csv',
        output_folder='./data/vit_features/',
        model_name='google/vit-base-patch16-224'
    )

    print("\n" + "="*60)
    print("EXAMPLE 3: Inspect extracted features")
    print("="*60)

    feature_file = './data/vit_features/001.001.001.001_expert.npy'
    if os.path.exists(feature_file):
        inspect_vit_features(feature_file)


def compare_vit_models():
    """Compare different ViT models for your use case."""

    models = [
        ('google/vit-base-patch16-224', 'Standard ViT-Base (ImageNet)'),
        ('google/vit-large-patch16-224', 'Larger ViT (better quality, slower)'),
        ('facebook/dino-vitb16', 'DINO (self-supervised, good for faces)'),
    ]

    # Note: You'd need to load a sample video/image to actually compare
    print("Recommended ViT models for facial expression:")
    print("\n1. google/vit-base-patch16-224")
    print("   - 768-dim embeddings")
    print("   - Good balance of speed/quality")
    print("   - Trained on ImageNet")
    print("   - Recommended for most cases")

    print("\n2. facebook/dino-vitb16")
    print("   - 768-dim embeddings")
    print("   - Self-supervised learning")
    print("   - Often better for faces/fine-grained features")
    print("   - Recommended for expression recognition")

    print("\n3. google/vit-large-patch16-224")
    print("   - 1024-dim embeddings")
    print("   - Higher quality, slower")
    print("   - Use if you have GPU and need best performance")


if __name__ == '__main__':
    # Check dependencies
    try:
        import transformers
        print("✓ transformers installed")
    except ImportError:
        print("Please install: pip install transformers")

    try:
        import cv2
        print("✓ opencv installed")
    except ImportError:
        print("Please install: pip install opencv-python")

    print("\n")
    compare_vit_models()

    # Uncomment to run extraction:
    # example_usage()
