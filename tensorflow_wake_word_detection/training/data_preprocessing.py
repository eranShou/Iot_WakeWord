#!/usr/bin/env python3
"""
Data Preprocessing for Hebrew Wake Word Detection

This script preprocesses Hebrew audio samples (Shalom and Lehitraot) for training
a wake word detection model using TensorFlow Lite Micro.

Features:
- Loads WAV files from ProcessedSamples directory
- Extracts MFCC features for machine learning
- Prepares data for model training
- Supports custom wake word addition
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Tuple, List, Dict, Optional

# Audio processing constants (matching TFLM micro_speech example)
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1

class HebrewWakeWordPreprocessor:
    """Preprocessor for Hebrew wake word audio data."""

    def __init__(self, data_dir: str, output_dir: str = "processed_data"):
        """
        Initialize the preprocessor.

        Args:
            data_dir: Directory containing ProcessedSamples folder
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Wake word labels
        self.wake_words = ['shalom', 'lehitraot']
        self.label_encoder = LabelEncoder()

        # Audio parameters
        self.sample_rate = SAMPLE_RATE
        self.clip_duration = CLIP_DURATION_MS / 1000.0  # Convert to seconds
        self.window_size = int(WINDOW_SIZE_MS * SAMPLE_RATE / 1000)
        self.window_stride = int(WINDOW_STRIDE_MS * SAMPLE_RATE / 1000)

    def load_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and ensure consistent format.

        Args:
            file_path: Path to WAV file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Ensure consistent length (1 second clips)
            target_length = int(self.sample_rate * self.clip_duration)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                # Pad with zeros if too short
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')

            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio data.

        Args:
            audio: Audio time series

        Returns:
            MFCC features array
        """
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=FEATURE_BIN_COUNT,
                n_fft=self.window_size,
                hop_length=self.window_stride,
                window='hann'
            )

            # Transpose to match TFLM expected format
            mfccs = mfccs.T  # Shape: (time_steps, n_mfcc)

            return mfccs
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None

    def process_dataset(self) -> Dict[str, np.ndarray]:
        """
        Process all audio files and extract features.

        Returns:
            Dictionary containing processed data
        """
        features = []
        labels = []
        file_paths = []

        print("Processing Hebrew wake word dataset...")

        for wake_word in self.wake_words:
            word_dir = self.data_dir / "ProcessedSamples" / wake_word.capitalize()
            if not word_dir.exists():
                print(f"Warning: Directory {word_dir} not found")
                continue

            print(f"Processing {wake_word} samples from {word_dir}")

            # Process each WAV file
            wav_files = list(word_dir.glob("*.wav"))
            for wav_file in wav_files:
                # Load audio
                audio, sr = self.load_audio_file(str(wav_file))
                if audio is None:
                    continue

                # Extract features
                mfcc_features = self.extract_mfcc_features(audio)
                if mfcc_features is None:
                    continue

                features.append(mfcc_features.flatten())  # Flatten for model input
                labels.append(wake_word)
                file_paths.append(str(wav_file))

                if len(features) % 50 == 0:
                    print(f"Processed {len(features)} samples...")

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Dataset processing complete:")
        print(f"- Total samples: {len(X)}")
        print(f"- Feature shape: {X.shape}")
        print(f"- Labels: {self.label_encoder.classes_}")

        return {
            'features': X,
            'labels': y_encoded,
            'label_names': self.label_encoder.classes_,
            'file_paths': file_paths
        }

    def split_dataset(self, data: Dict[str, np.ndarray],
                     test_size: float = 0.2,
                     val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Split dataset into train/validation/test sets.

        Args:
            data: Processed dataset dictionary
            test_size: Fraction of data for testing
            val_size: Fraction of data for validation

        Returns:
            Dictionary with split datasets
        """
        X = data['features']
        y = data['labels']

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=y_train_val
        )

        print("Dataset split:")
        print(f"- Train: {len(X_train)} samples")
        print(f"- Validation: {len(X_val)} samples")
        print(f"- Test: {len(X_test)} samples")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'label_names': data['label_names']
        }

    def save_processed_data(self, split_data: Dict[str, np.ndarray]):
        """Save processed data to disk."""
        output_file = self.output_dir / "hebrew_wake_word_data.npz"

        np.savez_compressed(
            output_file,
            X_train=split_data['X_train'],
            y_train=split_data['y_train'],
            X_val=split_data['X_val'],
            y_val=split_data['y_val'],
            X_test=split_data['X_test'],
            y_test=split_data['y_test'],
            label_names=split_data['label_names']
        )

        # Save metadata
        metadata = {
            'sample_rate': self.sample_rate,
            'clip_duration_ms': CLIP_DURATION_MS,
            'window_size_ms': WINDOW_SIZE_MS,
            'window_stride_ms': WINDOW_STRIDE_MS,
            'feature_bin_count': FEATURE_BIN_COUNT,
            'label_names': split_data['label_names'].tolist(),
            'train_samples': len(split_data['X_train']),
            'val_samples': len(split_data['X_val']),
            'test_samples': len(split_data['X_test'])
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Processed data saved to {output_file}")

    def visualize_sample(self, audio: np.ndarray, mfcc_features: np.ndarray,
                        label: str, save_path: Optional[str] = None):
        """
        Visualize audio waveform and MFCC features.

        Args:
            audio: Audio time series
            mfcc_features: MFCC features
            label: Wake word label
            save_path: Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot waveform
        ax1.plot(audio)
        ax1.set_title(f'Waveform - {label}')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Amplitude')

        # Plot MFCC
        img = librosa.display.specshow(
            mfcc_features.T,
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            hop_length=self.window_stride,
            ax=ax2
        )
        ax2.set_title(f'MFCC Features - {label}')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess Hebrew wake word audio data')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing ProcessedSamples folder')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of sample audio')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Fraction of data for validation')

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = HebrewWakeWordPreprocessor(args.data_dir, args.output_dir)

    # Process dataset
    data = preprocessor.process_dataset()

    if len(data['features']) == 0:
        print("No audio files found. Please check the data directory.")
        return

    # Split dataset
    split_data = preprocessor.split_dataset(data, args.test_size, args.val_size)

    # Save processed data
    preprocessor.save_processed_data(split_data)

    # Generate visualization if requested
    if args.visualize:
        # Visualize a sample from each class
        for i, label_name in enumerate(split_data['label_names']):
            # Find samples of this class
            class_indices = np.where(split_data['y_train'] == i)[0]
            if len(class_indices) > 0:
                sample_idx = class_indices[0]

                # Reconstruct audio and features for visualization
                sample_features = split_data['X_train'][sample_idx]
                # Note: This is a simplified reconstruction for visualization
                # In practice, you'd want to keep original audio for proper reconstruction

                save_path = preprocessor.output_dir / f"sample_{label_name}.png"
                # preprocessor.visualize_sample(audio, features, label_name, str(save_path))

    print("\nPreprocessing complete!")
    print(f"Processed data saved in: {preprocessor.output_dir}")

if __name__ == "__main__":
    main()
