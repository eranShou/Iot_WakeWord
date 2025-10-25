"""
Complete Dataset Processing for ESP32 Wake Word Detection
Conservative augmentation strategy that preserves ESP32 microphone characteristics.

Processing approach:
- Wake word directories (lehitraoot, shalom, bait): Apply augmentation to reach target sample count (2250)
- Unknown directory: Apply augmentation to reach target sample count (1000)
- Background directory: Copy as-is without augmentation
- Light time stretching: 0.9x-1.1x (±10% maximum) - simulates natural speech rate differences
- Background noise mixing: 25-35 dB SNR - very subtle noise at ESP32 level
- Preserves device-specific noise floor and frequency response characteristics
- Maintains -32 LUFS level as the device's natural signature

Avoids:
- Heavy pitch shifting (alters mic frequency response)
- Aggressive gain changes (destroys device signature)
- Artificial reverb/filters (destroys device characteristics)
"""

import os
import glob
import random
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import sys

# Import audiolib from same directory
from audiolib import audioread, audiowrite, snr_mixer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioAugmentor:
    """Main class for complete dataset processing including augmentation and copying."""
    
    def __init__(self, data_dir: str = "../database", noise_dir: str = "../database/background", target_samples: int = 2250):
        """
        Initialize the augmentor.
        
        Args:
            data_dir: Directory containing wake word audio files
            noise_dir: Directory containing noise files for mixing
            target_samples: Target number of samples per class (2000-2500 range)
        """
        self.data_dir = Path(data_dir)
        self.noise_dir = Path(noise_dir)
        self.target_samples = target_samples
        self.sample_rate = 16000
        
        # Conservative augmentation parameters for ESP32
        self.time_stretches = [0.9, 0.95, 1.0, 1.05, 1.1]  # ±10% maximum
        self.snr_levels = [25, 30, 35]  # very subtle noise (25-35 dB SNR)
        
        # Load noise files
        self.noise_files = self._load_noise_files()
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'augmentations_created': 0,
            'errors': 0,
            'start_time': None,
            'class_stats': {}
        }
    
    def _load_noise_files(self) -> List[Path]:
        """Load all available noise files."""
        if not self.noise_dir.exists():
            logger.warning(f"Noise directory not found: {self.noise_dir}")
            return []
        
        noise_files = list(self.noise_dir.glob("*.wav"))
        logger.info(f"Loaded {len(noise_files)} noise files")
        return noise_files
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and ensure correct sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, file_path: Path, sr: int = None) -> bool:
        """
        Save audio array to file.
        
        Args:
            audio: Audio array to save
            file_path: Output file path
            sr: Sample rate (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if sr is None:
                sr = self.sample_rate
            
            # Ensure output directory exists (especially important for augmented/ structure)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save using soundfile for better quality
            sf.write(str(file_path), audio, sr)
            logger.debug(f"Saved augmented audio: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio {file_path}: {e}")
            return False
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Apply time stretching using librosa.
        
        Args:
            audio: Input audio array
            rate: Stretch factor (>1 = slower, <1 = faster)
            
        Returns:
            Time-stretched audio array
        """
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            logger.error(f"Error in time stretch: {e}")
            return audio
    
    def add_noise(self, audio: np.ndarray, snr_db: float) -> Tuple[np.ndarray, str]:
        """
        Add background noise to audio at specified SNR.
        
        Args:
            audio: Input audio array
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Tuple of (noisy_audio, noise_type)
        """
        if not self.noise_files:
            logger.warning("No noise files available, returning original audio")
            return audio, "none"
        
        try:
            # Select random noise file
            noise_file = random.choice(self.noise_files)
            noise_type = noise_file.stem
            
            # Load noise
            noise, _ = librosa.load(str(noise_file), sr=self.sample_rate)
            
            # Ensure noise is at least as long as audio
            if len(noise) < len(audio):
                # Repeat noise if needed
                noise = np.tile(noise, (len(audio) // len(noise)) + 1)
            
            # Trim noise to match audio length
            noise = noise[:len(audio)]
            
            # Use SNR mixer from audiolib
            clean_snr, noise_snr, noisy_snr = snr_mixer(clean=audio, noise=noise, snr=snr_db)
            
            return noisy_snr, noise_type
        except Exception as e:
            logger.error(f"Error adding noise: {e}")
            return audio, "none"
    
    def create_augmented_filename(self, original_path: Path, augmentations: List[str], output_dir: Path) -> Path:
        """
        Create filename for augmented audio with descriptive suffixes.
        
        Args:
            original_path: Path to original file
            augmentations: List of augmentation descriptions
            output_dir: Output directory for augmented files
            
        Returns:
            New file path with augmentation suffixes in augmented_dataset/ directory
        """
        stem = original_path.stem
        suffix = "_" + "_".join(augmentations) if augmentations else ""
        
        # Create augmented directory structure: augmented_dataset/word_name/
        class_dir = output_dir / original_path.parent.name
        return class_dir / f"{stem}{suffix}.wav"
    
    def copy_original_file(self, original_path: Path, output_dir: Path) -> bool:
        """
        Copy original file to output directory.
        
        Args:
            original_path: Path to original file
            output_dir: Output directory for dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create class directory
            class_dir = output_dir / original_path.parent.name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy original file
            output_path = class_dir / original_path.name
            shutil.copy2(original_path, output_path)
            logger.debug(f"Copied original: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error copying original {original_path}: {e}")
            return False
    
    def _copy_directory_recursive(self, src_dir: Path, dest_dir: Path) -> None:
        """
        Recursively copy directory structure and all files.
        
        Args:
            src_dir: Source directory to copy
            dest_dir: Destination directory
        """
        try:
            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all files and subdirectories
            for item in src_dir.iterdir():
                if item.is_file():
                    dest_file = dest_dir / item.name
                    shutil.copy2(item, dest_file)
                elif item.is_dir():
                    dest_subdir = dest_dir / item.name
                    self._copy_directory_recursive(item, dest_subdir)
                    
            logger.debug(f"Recursively copied {src_dir} to {dest_dir}")
        except Exception as e:
            logger.error(f"Error copying directory {src_dir} to {dest_dir}: {e}")
            raise
    
    def calculate_augmentation_strategy(self, original_count: int, target_samples: int = None) -> dict:
        """
        Calculate augmentation strategy based on target samples and original count.
        
        Args:
            original_count: Number of original samples
            target_samples: Target number of samples (uses self.target_samples if None)
            
        Returns:
            Dictionary with augmentation parameters
        """
        if target_samples is None:
            target_samples = self.target_samples
        
        needed_augmentations = target_samples - original_count
        augmentations_per_file = needed_augmentations / original_count
        
        logger.info(f"Original samples: {original_count}, Target: {target_samples}")
        logger.info(f"Need {needed_augmentations} augmentations ({augmentations_per_file:.1f} per file)")
        
        # Determine strategy based on expansion ratio
        if augmentations_per_file > 50:  # High expansion (lehitraoot: ~68x)
            return {
                'time_stretches': self.time_stretches,  # All stretches
                'snr_levels': self.snr_levels,  # All SNR levels
                'combinations_per_stretch': 8,  # Many combinations
                'max_noise_types': 5  # Use many noise types
            }
        elif augmentations_per_file > 20:  # Medium expansion (shalom: ~31x)
            return {
                'time_stretches': [0.9, 0.95, 1.05, 1.1],  # Most stretches
                'snr_levels': [25, 30, 35],  # All SNR levels
                'combinations_per_stretch': 4,  # Moderate combinations
                'max_noise_types': 3  # Use moderate noise types
            }
        else:  # Low expansion (bait: ~3-4x)
            return {
                'time_stretches': [0.9, 0.95, 1.05, 1.1],  # Minimal stretches
                'snr_levels': [25],  # Only one SNR level
                'combinations_per_stretch': 1,  # Minimal combinations
                'max_noise_types': 1  # Use only one noise type
            }
    
    def augment_file(self, file_path: Path, strategy: dict, output_dir: Path) -> int:
        """
        Apply all augmentations to a single file.
        
        Args:
            file_path: Path to input audio file
            strategy: Augmentation strategy dictionary
            output_dir: Output directory for augmented files
            
        Returns:
            Number of augmented files created
        """
        logger.info(f"Processing: {file_path.name}")
        
        try:
            # Load original audio
            audio, sr = self.load_audio(file_path)
            
            augmentations_created = 0
            
            # 1. Independent augmentations
            augmentations_created += self._apply_independent_augmentations(file_path, audio, sr, strategy, output_dir)
            
            # 2. Combined augmentations
            augmentations_created += self._apply_combined_augmentations(file_path, audio, sr, strategy, output_dir)
            
            self.stats['files_processed'] += 1
            self.stats['augmentations_created'] += augmentations_created
            
            logger.info(f"Created {augmentations_created} augmentations for {file_path.name}")
            return augmentations_created
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['errors'] += 1
            return 0
    
    def _apply_independent_augmentations(self, file_path: Path, audio: np.ndarray, sr: int, strategy: dict, output_dir: Path) -> int:
        """Apply conservative independent augmentations for ESP32."""
        created = 0
        
        # Light time stretching
        for stretch in strategy['time_stretches']:
            if stretch == 1.0:  # Skip original
                continue
            augmented_audio = self.time_stretch(audio, stretch)
            output_path = self.create_augmented_filename(file_path, [f"stretch{stretch:.2f}"], output_dir)
            if self.save_audio(augmented_audio, output_path, sr):
                created += 1
        
        # Very subtle background noise mixing
        for snr in strategy['snr_levels']:
            augmented_audio, noise_type = self.add_noise(audio, snr)
            output_path = self.create_augmented_filename(file_path, [f"noise_{noise_type}_snr{snr}"], output_dir)
            if self.save_audio(augmented_audio, output_path, sr):
                created += 1
        
        return created
    
    def _apply_combined_augmentations(self, file_path: Path, audio: np.ndarray, sr: int, strategy: dict, output_dir: Path) -> int:
        """Apply conservative combined augmentations for ESP32 (time stretch + noise only)."""
        created = 0
        
        # Time stretch + Noise combinations (conservative approach)
        for stretch in strategy['time_stretches']:
            if stretch == 1.0:
                continue
            
            # Sample noise files based on strategy
            noise_samples = random.sample(self.noise_files, min(strategy['max_noise_types'], len(self.noise_files)))
            
            for snr in strategy['snr_levels']:
                for noise_file in noise_samples:
                    noise_type = noise_file.stem
                    
                    # Apply time stretch first, then noise
                    temp_audio = self.time_stretch(audio, stretch)
                    augmented_audio, _ = self.add_noise(temp_audio, snr)
                    
                    output_path = self.create_augmented_filename(
                        file_path, [f"stretch{stretch:.2f}", f"noise_{noise_type}_snr{snr}"], output_dir
                    )
                    if self.save_audio(augmented_audio, output_path, sr):
                        created += 1
                    
                    # Limit combinations based on strategy
                    if created >= strategy['combinations_per_stretch'] * len(strategy['time_stretches']):
                        break
                if created >= strategy['combinations_per_stretch'] * len(strategy['time_stretches']):
                    break
            if created >= strategy['combinations_per_stretch'] * len(strategy['time_stretches']):
                break
        
        return created
    
    def augment_dataset(self) -> dict:
        """
        Augment all audio files in the dataset.
        
        Returns:
            Dictionary with augmentation statistics
        """
        import time
        self.stats['start_time'] = time.time()
        
        # Create output directory in main project directory
        output_dir = Path("../augmented_dataset")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Starting audio augmentation...")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Augmented files will be saved to: {output_dir}")
        logger.info(f"Noise directory: {self.noise_dir}")
        logger.info(f"Available noise files: {len(self.noise_files)}")
        logger.info(f"Target samples per class: {self.target_samples}")
        
        # Copy background directory as-is (no augmentation)
        copy_dirs = ["background"]
        for copy_dir_name in copy_dirs:
            copy_dir = self.data_dir / copy_dir_name
            if copy_dir.exists():
                logger.info(f"\nCopying {copy_dir_name} directory as-is (no augmentation)")
                dest_dir = output_dir / copy_dir_name
                self._copy_directory_recursive(copy_dir, dest_dir)
                
                # Count files in copied directory
                wav_files = list(dest_dir.rglob("*.wav"))
                logger.info(f"Copied {len(wav_files)} files from {copy_dir_name}")
                
                # Track statistics
                self.stats['class_stats'][copy_dir_name] = {
                    'original_files': len(wav_files),
                    'copied_files': len(wav_files),
                    'augmentations_created': 0,
                    'total_files': len(wav_files),
                    'expansion_ratio': 1.0
                }
            else:
                logger.warning(f"{copy_dir_name} directory not found, skipping")
        
        # Find all word directories (exclude noise, augmented, and background directories)
        word_dirs = [d for d in self.data_dir.iterdir() 
                    if d.is_dir() and d.name not in ["noise", "augmented", "background"]]
        
        if not word_dirs:
            logger.error(f"No word directories found in {self.data_dir}")
            return self.stats
        
        logger.info(f"Found {len(word_dirs)} word directories: {[d.name for d in word_dirs]}")
        
        # Process each word directory
        for word_dir in word_dirs:
            logger.info(f"\nProcessing word: {word_dir.name}")
            
            # Find all WAV files in the directory
            wav_files = list(word_dir.glob("*.wav"))
            
            if not wav_files:
                logger.warning(f"No WAV files found in {word_dir.name}")
                continue
            
            logger.info(f"Found {len(wav_files)} WAV files")
            
            # Calculate augmentation strategy for this class
            # Special handling: unknown class uses target of 1000 instead of 2250
            if word_dir.name == "unknown":
                target_for_this_class = 1000
            else:
                target_for_this_class = self.target_samples
            
            strategy = self.calculate_augmentation_strategy(len(wav_files), target_for_this_class)
            
            # Copy original files first
            original_copied = 0
            for wav_file in sorted(wav_files):
                if self.copy_original_file(wav_file, output_dir):
                    original_copied += 1
            
            logger.info(f"Copied {original_copied} original files")
            
            # Process each file for augmentation
            class_augmentations = 0
            for wav_file in sorted(wav_files):
                augmentations = self.augment_file(wav_file, strategy, output_dir)
                class_augmentations += augmentations
            
            # Track class statistics
            total_class_files = original_copied + class_augmentations
            self.stats['class_stats'][word_dir.name] = {
                'original_files': len(wav_files),
                'copied_files': original_copied,
                'augmentations_created': class_augmentations,
                'total_files': total_class_files,
                'expansion_ratio': total_class_files / len(wav_files)
            }
            
            logger.info(f"Class {word_dir.name} completed: {total_class_files} total files ({total_class_files / len(wav_files):.1f}x expansion)")
        
        # Calculate final statistics
        end_time = time.time()
        processing_time = end_time - self.stats['start_time']
        
        self.stats['processing_time'] = processing_time
        self.stats['total_files'] = self.stats['files_processed'] + self.stats['augmentations_created']
        
        logger.info("\n" + "="*60)
        logger.info("AUGMENTATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Augmentations created: {self.stats['augmentations_created']}")
        logger.info(f"Total files: {self.stats['total_files']}")
        logger.info(f"Augmented files saved to: {output_dir}")
        logger.info(f"Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
        logger.info(f"Errors: {self.stats['errors']}")
        
        # Log class statistics
        logger.info("\nClass Statistics:")
        for class_name, class_stats in self.stats['class_stats'].items():
            logger.info(f"  {class_name}: {class_stats['total_files']} files ({class_stats['expansion_ratio']:.1f}x expansion)")
        
        return self.stats


def main():
    """Main function to run the complete dataset processing pipeline."""
    logger.info("="*60)
    logger.info("Complete Dataset Processing for Wake Word Detection")
    logger.info("="*60)
    
    # Initialize augmentor
    augmentor = AudioAugmentor(target_samples=2250)  # Target middle of 2000-2500 range
    
    # Run augmentation
    stats = augmentor.augment_dataset()
    
    # Print summary
    print(f"\nAugmentation Summary:")
    print(f"- Original files: {stats['files_processed']}")
    print(f"- Augmented files: {stats['augmentations_created']}")
    print(f"- Total files: {stats['total_files']}")
    print(f"- Processing time: {stats['processing_time']:.2f} seconds")
    
    # Print class summaries
    print(f"\nClass Details:")
    for class_name, class_stats in stats['class_stats'].items():
        print(f"- {class_name}: {class_stats['total_files']} files ({class_stats['expansion_ratio']:.1f}x expansion)")
    
    return stats


if __name__ == "__main__":
    main()
