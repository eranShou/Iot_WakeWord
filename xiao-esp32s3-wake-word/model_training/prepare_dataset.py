"""
Dataset preparation for Hebrew wake word detection
Loads 4 classes with proper train/validation split and converts to spectrograms
All parameters loaded from config.json - no magic numbers
"""

import json
import os
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def load_audio_files(directory, sample_rate, num_samples):
    """
    Load all WAV files from a directory and normalize to fixed length
    Returns: (audio_data, labels) where labels are class indices
    """
    audio_data = []
    labels = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return np.array([]), np.array([])
    
    wav_files = list(Path(directory).glob('*.wav'))
    print(f"Loading {len(wav_files)} files from {directory}")
    
    for wav_file in wav_files:
        try:
            # Load audio file
            audio, sr = librosa.load(str(wav_file), sr=sample_rate, mono=True)
            
            # Pad or truncate to exact length
            if len(audio) < num_samples:
                # Pad with zeros
                audio = np.pad(audio, (0, num_samples - len(audio)), mode='constant')
            elif len(audio) > num_samples:
                # Truncate from center
                start = (len(audio) - num_samples) // 2
                audio = audio[start:start + num_samples]
            
            audio_data.append(audio)
            
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
    
    return np.array(audio_data), np.array(labels)

def create_spectrogram(audio, frame_length, frame_step, fft_length, target_height, target_width):
    """
    Convert audio to spectrogram and resize to target dimensions
    """
    # Compute STFT
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )
    
    # Convert to magnitude spectrogram
    magnitude = tf.abs(stft)
    
    # Resize to target dimensions
    magnitude = tf.image.resize(magnitude[..., tf.newaxis], [target_height, target_width])
    
    return magnitude

def prepare_datasets():
    """
    Main function to prepare training and validation datasets
    Returns: (train_dataset, val_dataset, class_weights)
    """
    config = load_config()
    
    # Extract configuration
    audio_config = config['audio']
    spectrogram_config = config['spectrogram']
    training_config = config['training']
    data_paths = config['data_paths']
    classes = config['classes']
    
    print("Loading datasets from augmented_dataset...")
    print("=" * 50)
    
    # Load all data from augmented_dataset
    all_audio = []
    all_labels = []
    
    # Load each class and apply 80/20 split
    for class_name, class_path in data_paths.items():
        print(f"Loading {class_name} from {class_path}...")
        audio_data, _ = load_audio_files(
            class_path,
            audio_config['sample_rate'],
            audio_config['num_samples']
        )
        
        if len(audio_data) > 0:
            # Apply 80/20 train/val split
            train_data, val_data = train_test_split(
                audio_data,
                test_size=0.2,
                random_state=42
            )
            
            # Add to training data
            all_audio.extend(train_data)
            all_labels.extend([classes['label_map'][class_name]] * len(train_data))
            
            # Add to validation data
            all_audio.extend(val_data)
            all_labels.extend([classes['label_map'][class_name]] * len(val_data))
            
            print(f"  {class_name}: {len(train_data)} train, {len(val_data)} val")
        else:
            print(f"  {class_name}: No data found")
    
    # Convert to numpy arrays
    all_audio = np.array(all_audio)
    all_labels = np.array(all_labels)
    
    # Split into train and validation sets
    train_audio, val_audio, train_labels, val_labels = train_test_split(
        all_audio, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_audio)}")
    print(f"Validation samples: {len(val_audio)}")
    
    # Print class distribution
    for i, class_name in enumerate(classes['labels']):
        train_count = np.sum(train_labels == i)
        val_count = np.sum(val_labels == i)
        print(f"{class_name}: {train_count} train, {val_count} val")
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(classes['labels']))}
    
    print(f"\nClass weights: {class_weight_dict}")
    
    # Convert to spectrograms
    print("\nConverting to spectrograms...")
    
    def create_spectrograms(audio_data):
        spectrograms = []
        for audio in audio_data:
            spec = create_spectrogram(
                audio,
                spectrogram_config['frame_length'],
                spectrogram_config['frame_step'],
                spectrogram_config['fft_length'],
                spectrogram_config['target_height'],
                spectrogram_config['target_width']
            )
            spectrograms.append(spec)
        return np.array(spectrograms)
    
    train_spectrograms = create_spectrograms(train_audio)
    val_spectrograms = create_spectrograms(val_audio)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_spectrograms, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_spectrograms, val_labels))
    
    # Configure datasets
    train_dataset = train_dataset.batch(training_config['batch_size']).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(training_config['batch_size']).cache().prefetch(tf.data.AUTOTUNE)
    
    print(f"\nDataset preparation complete!")
    print(f"Training batches: {len(train_dataset)}")
    print(f"Validation batches: {len(val_dataset)}")
    
    return train_dataset, val_dataset, class_weight_dict

if __name__ == "__main__":
    train_ds, val_ds, class_weights = prepare_datasets()
    print("Dataset preparation completed successfully!")
