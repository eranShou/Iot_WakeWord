"""
Audio Split Script
Splits WAV recordings by speech activity detection using auditok.
Each recording filename indicates the number of words it contains.
"""

import os
import shutil
from pathlib import Path
import auditok
import re
import random
import wave
import numpy as np
import tempfile

def get_word_count_from_filename(filename):
    """
    Extract word count from filename.
    Filename format: {number}.wav (e.g., '10.wav' = 10 words)
    """
    # Extract number from filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def is_segment_silent(audio_data, sample_rate, silence_threshold=0.01):
    """
    Check if an audio segment is completely silent.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        silence_threshold: Energy threshold below which audio is considered silent
    
    Returns:
        bool: True if segment is silent, False otherwise
    """
    if len(audio_data) == 0:
        return True
    
    # Calculate RMS (Root Mean Square) energy
    rms_energy = np.sqrt(np.mean(audio_data**2))
    
    return rms_energy < silence_threshold

def process_segment_to_one_second(region, sample_rate=16000):
    """
    Process a speech region to exactly 1 second.
    
    Args:
        region: auditok AudioRegion object
        sample_rate: Target sample rate (default 16000 Hz)
    
    Returns:
        tuple: (processed_audio_data, is_silent) or (None, True) if silent
    """
    # Save the region to a temporary file and read it back as numpy array
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Save the region to temporary file
        region.save(temp_path)
        
        # Read the audio file using wave module
        with wave.open(temp_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            # Convert bytes to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        duration = region.duration
        
        # Calculate target samples for 1 second
        target_samples = sample_rate  # 1 second at sample_rate
        
        if duration < 1.0:
            # Pad with silence to reach 1 second
            current_samples = len(audio_data)
            padding_samples = target_samples - current_samples
            if padding_samples > 0:
                # Pad with zeros (silence)
                padding = np.zeros(padding_samples, dtype=audio_data.dtype)
                audio_data = np.concatenate([audio_data, padding])
        
        elif duration > 1.0:
            # Randomly choose first or last second
            current_samples = len(audio_data)
            if current_samples > target_samples:
                if random.choice([True, False]):
                    # Keep first second
                    audio_data = audio_data[:target_samples]
                else:
                    # Keep last second
                    audio_data = audio_data[-target_samples:]
        
        # Check if the resulting segment is silent
        is_silent = is_segment_silent(audio_data, sample_rate)
        
        if is_silent:
            return None, True
        else:
            return audio_data, False
            
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def split_audio_file(input_file, output_dir, word_name, expected_words, file_counter):
    """
    Split a single audio file using auditok speech activity detection.
    
    Args:
        input_file: Path to input WAV file
        output_dir: Directory to save split segments
        word_name: Name of the word being processed
        expected_words: Expected number of words in the recording
        file_counter: Starting counter for output files
    
    Returns:
        tuple: (success, segments_created, next_counter)
    """
    try:
        print(f"  Loading audio: {input_file.name}")
        
        # Use auditok to detect speech activity and split audio
        print(f"  Detecting speech activity with auditok...")
        
        # Configure auditok parameters for speech detection
        # Use more sensitive parameters for short silences between words
        audio_regions = auditok.split(
            str(input_file),
            min_dur=0.1,      # Minimum duration of speech segment (100ms)
            max_dur=2.0,       # Maximum duration of speech segment (2s) - shorter for individual words
            max_silence=0.1,   # Maximum silence between words (100ms) - very short
            energy_threshold=35  # Lower energy threshold for better sensitivity
        )
        
        segments_created = 0
        silent_segments_skipped = 0
        
        # Process each detected speech region to exactly 1 second
        for i, region in enumerate(audio_regions):
            # Process the region to exactly 1 second
            processed_audio, is_silent = process_segment_to_one_second(region)
            
            if is_silent:
                print(f"    Skipping silent segment {i+1} (duration: {region.duration:.2f}s)")
                silent_segments_skipped += 1
                continue
            
            # Save the processed 1-second segment
            segment_filename = f"{word_name}_{file_counter + segments_created:03d}.wav"
            segment_path = output_dir / segment_filename
            
            # Save the processed audio data as WAV file
            try:
                # Convert numpy array to WAV file
                with wave.open(str(segment_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz sample rate
                    wav_file.writeframes((processed_audio * 32767).astype(np.int16).tobytes())
                
                print(f"    Saving 1-second segment {segments_created + 1}: {segment_filename}")
                segments_created += 1
                
            except Exception as e:
                print(f"    ERROR saving segment {i+1}: {e}")
                continue
        
        print(f"  Created {segments_created} 1-second segments (expected: {expected_words})")
        if silent_segments_skipped > 0:
            print(f"  Skipped {silent_segments_skipped} silent segments")
        
        # Check if segment count matches expected words (±1 margin)
        if abs(segments_created - expected_words) > 1:
            print(f"  WARNING: Segment count ({segments_created}) doesn't match expected words ({expected_words}) ±1")
        
        return True, segments_created, file_counter + segments_created
        
    except Exception as e:
        print(f"  ERROR processing {input_file.name}: {e}")
        return False, 0, file_counter

def split_audio_recordings():
    """
    Split all audio recordings in the data directory using auditok speech activity detection.
    First processes files from main word folders, then moves originals to backup.
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    
    print(f"Processing audio files in: {data_dir}")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please run copy_recordings.py first to set up the data structure.")
        return False
    
    # Get all subfolders in data directory
    subfolders = [f for f in data_dir.iterdir() if f.is_dir() and f.name != "noise"]
    
    if not subfolders:
        print("No word folders found in data directory")
        return False
    
    print(f"Found {len(subfolders)} word folders: {[f.name for f in subfolders]}")
    
    total_segments_created = 0
    total_files_processed = 0
    total_files_moved = 0
    
    for subfolder in subfolders:
        word_name = subfolder.name
        print(f"\nProcessing word: {word_name}")
        
        # Find all WAV files in the main word folder (not backup)
        wav_files = list(subfolder.glob("*.wav"))
        
        if not wav_files:
            print(f"  No WAV files found in {word_name}")
            continue
        
        print(f"  Found {len(wav_files)} WAV files")
        
        # Sort files by name to maintain order
        wav_files.sort(key=lambda x: x.name)
        
        file_counter = 1  # Start numbering from 001
        
        # Create backup directory if it doesn't exist
        backup_dir = subfolder / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        for wav_file in wav_files:
            # Get expected word count from filename
            expected_words = get_word_count_from_filename(wav_file.name)
            
            if expected_words is None:
                print(f"  WARNING: Could not extract word count from {wav_file.name}, skipping")
                continue
            
            print(f"  Processing {wav_file.name} (expected {expected_words} words)")
            
            # Split the audio file
            success, segments_created, next_counter = split_audio_file(
                wav_file, subfolder, word_name, expected_words, file_counter
            )
            
            if success:
                total_segments_created += segments_created
                total_files_processed += 1
                file_counter = next_counter
                
                # Move original file to backup folder after successful splitting
                backup_file_path = backup_dir / wav_file.name
                try:
                    shutil.move(str(wav_file), str(backup_file_path))
                    print(f"  Moved original file to backup: {wav_file.name}")
                    total_files_moved += 1
                except Exception as e:
                    print(f"  WARNING: Could not move {wav_file.name} to backup: {e}")
            else:
                print(f"  Failed to process {wav_file.name}")
    
    print(f"\nSplit operation completed!")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total 1-second segments created: {total_segments_created}")
    print(f"Total files moved to backup: {total_files_moved}")
    print(f"All segments are exactly 1 second long (silent segments removed)")
    
    return True

def main():
    """Main function to run the audio splitting operation."""
    print("=" * 60)
    print("Audio Split Script - Speech Activity Detection (auditok)")
    print("=" * 60)
    
    success = split_audio_recordings()
    
    if success:
        print("\n[SUCCESS] Audio splitting completed successfully!")
        print("\nOutput structure:")
        print("- Original recordings moved to backup/ folders after processing")
        print("- All segments are exactly 1 second long")
        print("- Silent segments are automatically removed")
        print("- Segments saved as {word_name}_001.wav, {word_name}_002.wav, etc.")
        print("- Segment count should match filename number ±1")
    else:
        print("\n[ERROR] Audio splitting failed!")
        print("Make sure you have run copy_recordings.py first.")
    
    return success

if __name__ == "__main__":
    main()
