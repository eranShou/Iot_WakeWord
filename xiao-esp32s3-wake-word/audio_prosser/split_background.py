"""
Background Audio Split Script
Splits background WAV recordings into non-overlapping 1-second segments.
Outputs segments to database/background/ directory.
"""

import os
import wave
import numpy as np
from pathlib import Path
import shutil

def split_background_recordings():
    """
    Split background audio recordings into 1-second segments.
    Reads from audio_recorder_esp32_pc/recordings/background/
    Outputs to database/background/
    """
    # Define paths
    current_dir = Path(__file__).parent
    input_dir = current_dir.parent / "audio_recorder_esp32_pc" / "recordings" / "background"
    output_dir = current_dir.parent / "database" / "background"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files in input directory
    wav_files = list(input_dir.glob("*.wav"))
    
    if not wav_files:
        print("No WAV files found in input directory")
        return False
    
    print(f"Found {len(wav_files)} WAV files to process")
    
    # Sort files by name for consistent ordering
    wav_files.sort(key=lambda x: x.name)
    
    total_segments_created = 0
    segment_counter = 1
    
    for wav_file in wav_files:
        print(f"\nProcessing: {wav_file.name}")
        
        try:
            # Open WAV file
            with wave.open(str(wav_file), 'rb') as wav_reader:
                # Get audio parameters
                sample_rate = wav_reader.getframerate()
                n_channels = wav_reader.getnchannels()
                sampwidth = wav_reader.getsampwidth()
                n_frames = wav_reader.getnframes()
                
                print(f"  Sample rate: {sample_rate} Hz")
                print(f"  Channels: {n_channels}")
                print(f"  Sample width: {sampwidth} bytes")
                print(f"  Duration: {n_frames / sample_rate:.2f} seconds")
                
                # Read all audio data
                audio_data = wav_reader.readframes(n_frames)
                
                # Convert to numpy array
                if sampwidth == 2:  # 16-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                elif sampwidth == 1:  # 8-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                else:
                    print(f"  ERROR: Unsupported sample width: {sampwidth}")
                    continue
                
                # Convert to mono if stereo
                if n_channels == 2:
                    # Take left channel (every other sample)
                    audio_array = audio_array[::2]
                
                # Calculate samples per 1-second segment
                samples_per_segment = sample_rate  # 1 second worth of samples
                
                # Split into 1-second segments
                segments_created = 0
                start_sample = 0
                
                while start_sample + samples_per_segment <= len(audio_array):
                    # Extract 1-second segment
                    end_sample = start_sample + samples_per_segment
                    segment_data = audio_array[start_sample:end_sample]
                    
                    # Create output filename
                    output_filename = f"background_{segment_counter:03d}.wav"
                    output_path = output_dir / output_filename
                    
                    # Save segment as WAV file
                    with wave.open(str(output_path), 'wb') as wav_writer:
                        wav_writer.setnchannels(1)  # Mono
                        wav_writer.setsampwidth(2)  # 16-bit
                        wav_writer.setframerate(16000)  # 16kHz
                        
                        # Convert back to bytes and write
                        if sampwidth == 2:
                            wav_writer.writeframes(segment_data.tobytes())
                        else:
                            # Convert 8-bit to 16-bit
                            segment_16bit = (segment_data.astype(np.int16) - 128) * 256
                            wav_writer.writeframes(segment_16bit.tobytes())
                    
                    print(f"  Created: {output_filename}")
                    segments_created += 1
                    segment_counter += 1
                    total_segments_created += 1
                    
                    # Move to next segment
                    start_sample += samples_per_segment
                
                # Check if there's remaining audio shorter than 1 second
                remaining_samples = len(audio_array) - start_sample
                if remaining_samples > 0:
                    print(f"  Discarding {remaining_samples} samples ({remaining_samples/sample_rate:.2f}s) - shorter than 1 second")
                
                print(f"  Created {segments_created} segments from {wav_file.name}")
                
        except Exception as e:
            print(f"  ERROR processing {wav_file.name}: {e}")
            continue
    
    print(f"\nSplit operation completed!")
    print(f"Total 1-second segments created: {total_segments_created}")
    print(f"Output directory: {output_dir}")
    
    return True

def main():
    """Main function to run the background audio splitting operation."""
    print("=" * 60)
    print("Background Audio Split Script")
    print("=" * 60)
    
    success = split_background_recordings()
    
    if success:
        print("\n[SUCCESS] Background audio splitting completed successfully!")
        print("\nOutput structure:")
        print("- All segments are exactly 1 second long")
        print("- Segments saved as background_001.wav, background_002.wav, etc.")
        print("- Segments saved in database/background/ directory")
        print("- Any remaining audio shorter than 1 second is discarded")
    else:
        print("\n[ERROR] Background audio splitting failed!")
    
    return success

if __name__ == "__main__":
    main()
