#!/usr/bin/env python3
"""
Custom Wake Word Training for Hebrew Wake Word Detection

This script allows users to easily add custom wake words to the existing
Hebrew wake word detection system by retraining the model with new audio samples.

Features:
- Add new wake words without retraining from scratch
- Transfer learning from pre-trained Hebrew model
- Easy audio sample collection and preprocessing
- Model update and conversion pipeline
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
from pathlib import Path
import librosa
import json
from typing import List, Dict, Optional, Tuple
import shutil
from sklearn.model_selection import train_test_split

# Import our existing modules
from data_preprocessing import HebrewWakeWordPreprocessor
from model_training import HebrewWakeWordModel

class CustomWakeWordTrainer:
    """Trainer for adding custom wake words to existing model."""

    def __init__(self, base_model_path: str, new_words: List[str],
                 custom_audio_dir: str, output_dir: str = "custom_models"):
        """
        Initialize custom word trainer.

        Args:
            base_model_path: Path to pre-trained base model
            new_words: List of new wake words to add
            custom_audio_dir: Directory containing custom audio samples
            output_dir: Output directory for updated models
        """
        self.base_model_path = Path(base_model_path)
        self.new_words = new_words
        self.custom_audio_dir = Path(custom_audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load base model
        self.base_model = keras.models.load_model(base_model_path)
        print(f"Base model loaded from {base_model_path}")

        # Get original model info
        self.original_classes = self._load_model_classes()
        print(f"Original classes: {self.original_classes}")

    def _load_model_classes(self) -> List[str]:
        """Load class names from model metadata."""
        # Try to find metadata file
        metadata_path = self.base_model_path.parent / "training_results_cnn.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                # This is a simplified assumption - in practice you'd store class names properly
                return ["shalom", "lehitraot"]  # Default Hebrew words

        # Fallback: infer from model output shape
        return [f"class_{i}" for i in range(self.base_model.output_shape[1])]

    def collect_custom_samples(self) -> Dict[str, List[np.ndarray]]:
        """
        Collect and preprocess custom audio samples.

        Returns:
            Dictionary mapping word names to list of audio features
        """
        custom_samples = {}

        for word in self.new_words:
            word_dir = self.custom_audio_dir / word
            if not word_dir.exists():
                print(f"Warning: Directory {word_dir} not found for word '{word}'")
                continue

            print(f"Processing samples for '{word}' from {word_dir}")

            features = []
            wav_files = list(word_dir.glob("*.wav"))

            if len(wav_files) == 0:
                print(f"Warning: No WAV files found in {word_dir}")
                continue

            preprocessor = HebrewWakeWordPreprocessor(".", "temp")

            for wav_file in wav_files:
                # Load and preprocess audio
                audio, sr = preprocessor.load_audio_file(str(wav_file))
                if audio is None:
                    continue

                # Extract features
                mfcc_features = preprocessor.extract_mfcc_features(audio)
                if mfcc_features is None:
                    continue

                features.append(mfcc_features.flatten())

                if len(features) % 10 == 0:
                    print(f"Processed {len(features)} samples for '{word}'...")

            if len(features) > 0:
                custom_samples[word] = features
                print(f"Collected {len(features)} samples for '{word}'")
            else:
                print(f"No valid samples found for '{word}'")

        # Cleanup
        if Path("temp").exists():
            shutil.rmtree("temp")

        return custom_samples

    def prepare_expanded_dataset(self, custom_samples: Dict[str, List[np.ndarray]],
                               original_data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare expanded dataset with original and custom samples.

        Args:
            custom_samples: Custom word samples
            original_data_path: Path to original processed data

        Returns:
            Tuple of (features, labels, class_names)
        """
        # Load original data
        try:
            data = np.load(original_data_path)
            X_original = data['X_train']
            y_original = data['y_train']
            original_classes = data['label_names']
        except FileNotFoundError:
            print(f"Original data not found at {original_data_path}")
            print("Using only custom samples...")
            X_original = np.array([])
            y_original = np.array([])
            original_classes = []

        # Prepare expanded dataset
        all_features = []
        all_labels = []
        all_classes = list(original_classes)

        # Add original samples
        if len(X_original) > 0:
            for i, features in enumerate(X_original):
                all_features.append(features)
                all_labels.append(y_original[i])

        # Add custom samples
        for word, features_list in custom_samples.items():
            if word not in all_classes:
                all_classes.append(word)

            word_label = all_classes.index(word)

            for features in features_list:
                all_features.append(features)
                all_labels.append(word_label)

        X_expanded = np.array(all_features)
        y_expanded = np.array(all_labels)

        print("Expanded dataset:")
        print(f"- Total samples: {len(X_expanded)}")
        print(f"- Classes: {all_classes}")
        print(f"- Class distribution: {np.bincount(y_expanded)}")

        return X_expanded, y_expanded, all_classes

    def expand_model_architecture(self, new_num_classes: int) -> keras.Model:
        """
        Expand model architecture to accommodate new classes.

        Args:
            new_num_classes: New total number of classes

        Returns:
            Expanded model
        """
        # Load base model
        model = keras.models.load_model(self.base_model_path)

        # Remove output layer
        model.layers.pop()
        model = keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)

        # Add new output layer
        x = model.output
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(new_num_classes, activation='softmax')(x)

        # Create new model
        expanded_model = keras.Model(inputs=model.inputs, outputs=outputs)

        # Freeze some layers for transfer learning
        for layer in expanded_model.layers[:-3]:  # Freeze all but last 3 layers
            layer.trainable = False

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # Lower learning rate
        expanded_model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Model expanded for transfer learning:")
        print(f"- Original classes: {self.base_model.output_shape[1]}")
        print(f"- New classes: {new_num_classes}")
        print(f"- Trainable layers: {sum([1 for layer in expanded_model.layers if layer.trainable])}")

        return expanded_model

    def retrain_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     expanded_model: keras.Model) -> keras.callbacks.History:
        """
        Retrain the expanded model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            expanded_model: Expanded model to train

        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_custom_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        print("Retraining model with custom words...")
        history = expanded_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,  # More epochs for transfer learning
            batch_size=16,  # Smaller batch size
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_updated_model_info(self, class_names: List[str], history: keras.callbacks.History):
        """Save information about the updated model."""
        model_info = {
            'original_model': str(self.base_model_path),
            'new_words_added': self.new_words,
            'all_classes': class_names,
            'num_classes': len(class_names),
            'training_history': {
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['accuracy'])
            }
        }

        info_path = self.output_dir / "custom_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"Custom model info saved to {info_path}")

def main():
    """Main custom training function."""
    parser = argparse.ArgumentParser(description='Add custom wake words to Hebrew detection model')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Path to base trained model (.h5)')
    parser.add_argument('--new_words', type=str, nargs='+', required=True,
                       help='New wake words to add (space-separated)')
    parser.add_argument('--custom_audio_dir', type=str, required=True,
                       help='Directory containing custom audio samples')
    parser.add_argument('--original_data', type=str, default='processed_data/hebrew_wake_word_data.npz',
                       help='Path to original processed data')
    parser.add_argument('--output_dir', type=str, default='custom_models',
                       help='Output directory for updated models')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Fraction of custom data for testing')

    args = parser.parse_args()

    print("=== Custom Wake Word Training ===")
    print(f"Base model: {args.base_model}")
    print(f"New words: {args.new_words}")
    print(f"Custom audio directory: {args.custom_audio_dir}")

    # Initialize trainer
    trainer = CustomWakeWordTrainer(
        args.base_model,
        args.new_words,
        args.custom_audio_dir,
        args.output_dir
    )

    # Collect custom samples
    print("\n1. Collecting custom audio samples...")
    custom_samples = trainer.collect_custom_samples()

    if not custom_samples:
        print("No custom samples found. Please check your audio directory structure.")
        return

    # Prepare expanded dataset
    print("\n2. Preparing expanded dataset...")
    X_expanded, y_expanded, class_names = trainer.prepare_expanded_dataset(
        custom_samples, args.original_data
    )

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_expanded, y_expanded, test_size=args.test_split * 2, random_state=42, stratify=y_expanded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Expand model architecture
    print("\n3. Expanding model architecture...")
    expanded_model = trainer.expand_model_architecture(len(class_names))

    # Retrain model
    print("\n4. Retraining model...")
    history = trainer.retrain_model(X_train, y_train, X_val, y_val, expanded_model)

    # Save updated model
    final_model_path = trainer.output_dir / "custom_wake_word_model.h5"
    expanded_model.save(final_model_path)
    print(f"Updated model saved to {final_model_path}")

    # Save model info
    trainer.save_updated_model_info(class_names, history)

    # Evaluate final model
    print("\n5. Evaluating updated model...")
    test_loss, test_accuracy = expanded_model.evaluate(X_test, y_test, verbose=0)
    print(".4f")
    print(".4f")

    print("\n=== Custom Training Complete ===")
    print(f"New model supports {len(class_names)} wake words:")
    for i, word in enumerate(class_names):
        marker = " (NEW)" if word in args.new_words else ""
        print(f"  {i}: {word}{marker}")

    print("Next steps:")
    print("1. Run model_conversion.py to convert to TFLite/TFLite Micro format")
    print("2. Update microcontroller code with new model and class labels")
    print("3. Test the updated wake word detection system")

if __name__ == "__main__":
    main()
