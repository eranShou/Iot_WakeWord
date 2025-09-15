#!/usr/bin/env python3
"""
Hebrew Wake Word Detection Model Training

This script trains a TensorFlow/Keras model for detecting Hebrew wake words
(Shalom and Lehitraot) using MFCC features extracted from audio data.

Based on the TensorFlow Lite Micro micro_speech example architecture.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, Dict, Optional

# Model constants (matching TFLM micro_speech)
FEATURE_BIN_COUNT = 40
WINDOW_STRIDE = 20
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10

class HebrewWakeWordModel:
    """TensorFlow model for Hebrew wake word detection."""

    def __init__(self, num_classes: int, input_shape: Tuple[int, ...]):
        """
        Initialize the model.

        Args:
            num_classes: Number of wake word classes
            input_shape: Shape of input features
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_model(self) -> keras.Model:
        """
        Build the CNN model based on tiny_conv architecture from TFLM.

        Returns:
            Compiled Keras model
        """
        # Calculate expected input size for MFCC features
        # 49 time frames * 40 frequency bins = 1960 features
        expected_input_size = 49 * FEATURE_BIN_COUNT

        # Validate input shape
        if self.input_shape[0] != expected_input_size:
            print(f"Warning: Input shape {self.input_shape} doesn't match expected shape ({expected_input_size},)")
            print(f"Expected: {expected_input_size} features (49 frames * 40 bins)")
            print(f"Got: {self.input_shape[0]} features")

            # If input is larger, we'll truncate; if smaller, we'll pad
            if self.input_shape[0] > expected_input_size:
                print(f"Input will be truncated to {expected_input_size} features")
            elif self.input_shape[0] < expected_input_size:
                print(f"Input will be padded to {expected_input_size} features")

        # Use Functional API to handle variable input shapes
        input_layer = keras.Input(shape=self.input_shape)

        # Handle input size mismatch using Dense layers
        if self.input_shape[0] != expected_input_size:
            if self.input_shape[0] > expected_input_size:
                # Reduce dimensionality to expected size
                x = layers.Dense(expected_input_size)(input_layer)
            else:
                # Expand dimensionality to expected size
                x = layers.Dense(expected_input_size)(input_layer)
        else:
            x = input_layer

        # Reshape to 2D for CNN
        x = layers.Reshape((49, FEATURE_BIN_COUNT, 1))(x)

        # First convolutional layer
        x = layers.Conv2D(
            filters=8,
            kernel_size=(10, 8),
            strides=(2, 2),
            activation='relu',
            padding='same'
        )(x)
        x = layers.Dropout(0.5)(x)

        # Second convolutional layer
        x = layers.Conv2D(
            filters=16,
            kernel_size=(5, 4),
            strides=(2, 2),
            activation='relu',
            padding='same'
        )(x)
        x = layers.Dropout(0.5)(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=input_layer, outputs=outputs)

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def build_simplified_model(self) -> keras.Model:
        """
        Build a simplified model for resource-constrained devices.

        Returns:
            Compiled simplified Keras model
        """
        model = keras.Sequential([
            layers.Flatten(input_shape=self.input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
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

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.history = history
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_names: np.ndarray) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            label_names: Label names for reporting

        Returns:
            Dictionary with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }

        return results

    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, label_names: np.ndarray,
                             save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

def load_processed_data(data_path: str) -> Tuple[np.ndarray, ...]:
    """
    Load processed data from npz file.

    Args:
        data_path: Path to processed data file

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, label_names)
    """
    data = np.load(data_path)

    return (
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['X_test'],
        data['y_test'],
        data['label_names']
    )

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Hebrew wake word detection model')
    parser.add_argument('--data_path', type=str, default='processed_data/hebrew_wake_word_data.npz',
                       help='Path to processed data file')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'simple'], default='cnn',
                       help='Model architecture type')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load processed data
    print(f"Loading data from {args.data_path}")
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_processed_data(args.data_path)
    except FileNotFoundError:
        print(f"Error: Data file {args.data_path} not found.")
        print("Please run data_preprocessing.py first.")
        return

    print(f"Data loaded successfully:")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Validation samples: {len(X_val)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Classes: {label_names}")
    print(f"- Feature shape: {X_train.shape[1:]}")

    # Initialize model
    num_classes = len(label_names)
    input_shape = X_train.shape[1:]

    model_trainer = HebrewWakeWordModel(num_classes, input_shape)

    # Build model
    if args.model_type == 'cnn':
        print("Building CNN model...")
        model_trainer.build_model()
    else:
        print("Building simplified model...")
        model_trainer.build_simplified_model()

    print(f"Model summary:")
    model_trainer.model.summary()

    # Train model
    print("\nStarting training...")
    history = model_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate model
    print("\nEvaluating model...")
    results = model_trainer.evaluate(X_test, y_test, label_names)

    print("Test Results:")
    print(f"- Loss: {results['test_loss']:.4f}")
    print(f"- Accuracy: {results['test_accuracy']:.4f}")

    # Save model
    model_path = output_dir / f"hebrew_wake_word_model_{args.model_type}.h5"
    model_trainer.save_model(str(model_path))

    # Save training history and results
    results_path = output_dir / f"training_results_{args.model_type}.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'test_loss': float(results['test_loss']),
            'test_accuracy': float(results['test_accuracy']),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        json.dump(json_results, f, indent=2)

    # Generate plots
    if model_trainer.history:
        plot_path = output_dir / f"training_history_{args.model_type}.png"
        model_trainer.plot_training_history(str(plot_path))

    cm_plot_path = output_dir / f"confusion_matrix_{args.model_type}.png"
    model_trainer.plot_confusion_matrix(results['confusion_matrix'], label_names, str(cm_plot_path))

    print("Training complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print(f"Plots saved in: {output_dir}")

if __name__ == "__main__":
    main()
