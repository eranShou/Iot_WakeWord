"""
Model training for Hebrew wake word detection
Config-driven CNN architecture with class-weighted training
All parameters loaded from config.json - no magic numbers
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from prepare_dataset import prepare_datasets

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def create_model(config):
    """
    Create CNN model based on configuration
    All architecture parameters from config.json
    """
    model_config = config['model']
    input_shape = model_config['input_shape']
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(
            model_config['conv1_filters'],
            model_config['conv1_kernel'],
            activation='relu',
            padding='same'
        ),
        
        # Second conv block
        layers.Conv2D(
            model_config['conv2_filters'],
            model_config['conv2_kernel'],
            activation='relu',
            padding='same'
        ),
        
        # Pooling
        layers.MaxPooling2D(pool_size=model_config['pool_size']),
        
        # First dropout
        layers.Dropout(model_config['dropout1_rate']),
        
        # Flatten
        layers.Flatten(),
        
        # Dense layer
        layers.Dense(model_config['dense_units'], activation='relu'),
        
        # Second dropout
        layers.Dropout(model_config['dropout2_rate']),
        
        # Output layer (no activation for logits)
        layers.Dense(model_config['num_classes'])
    ])
    
    return model

def train_model():
    """
    Main training function
    """
    config = load_config()
    
    print("Starting model training...")
    print("=" * 50)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, class_weights = prepare_datasets()
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Compile model
    training_config = config['training']
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_config['learning_rate']),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print(f"Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=training_config['early_stopping_patience'],
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            config['output']['keras_model'],
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nTraining for {training_config['epochs']} epochs...")
    print(f"Class weights: {class_weights}")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=training_config['epochs'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    
    # Generate predictions for detailed analysis
    print("\nGenerating predictions for analysis...")
    val_predictions = model.predict(val_dataset)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    # Get true labels
    val_labels = []
    for _, labels in val_dataset:
        val_labels.extend(labels.numpy())
    val_labels = np.array(val_labels)
    
    # Classification report
    classes = config['classes']['labels']
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_pred_classes, target_names=classes))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_pred_classes)
    print(cm)
    
    # Plot training history
    print("\nGenerating training plots...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(config['output']['training_history'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(config['output']['confusion_matrix'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i, class_name in enumerate(classes):
        class_mask = val_labels == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(val_pred_classes[class_mask] == i)
            print(f"{class_name}: {class_accuracy:.4f}")
        else:
            print(f"{class_name}: No samples in validation set")
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {config['output']['keras_model']}")
    print(f"Training history saved to: {config['output']['training_history']}")
    print(f"Confusion matrix saved to: {config['output']['confusion_matrix']}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    print("Training completed successfully!")
