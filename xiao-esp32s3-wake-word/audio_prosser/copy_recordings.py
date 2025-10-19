"""
Copy Recordings Script
Copies WAV files from audio_recorder_esp32_pc/recordings/ to audio_prosser/data/
Creates backup folders and organizes files by word name.
"""

import os
import shutil
from pathlib import Path

def copy_recordings():
    """Copy recordings from recordings/ folder to data/ folder with backup structure."""
    
    # Define paths
    current_dir = Path(__file__).parent
    recordings_dir = current_dir.parent / "audio_recorder_esp32_pc" / "recordings"
    data_dir = current_dir / "data"
    
    print(f"Source recordings directory: {recordings_dir}")
    print(f"Target data directory: {data_dir}")
    
    # Check if recordings directory exists
    if not recordings_dir.exists():
        print(f"Error: Recordings directory not found at {recordings_dir}")
        return False
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Get all subfolders in recordings directory
    subfolders = [f for f in recordings_dir.iterdir() if f.is_dir()]
    
    if not subfolders:
        print("No subfolders found in recordings directory")
        return False
    
    print(f"Found {len(subfolders)} subfolders: {[f.name for f in subfolders]}")
    
    total_files_copied = 0
    
    for subfolder in subfolders:
        word_name = subfolder.name
        print(f"\nProcessing folder: {word_name}")
        
        # Create target folder structure
        target_folder = data_dir / word_name
        backup_folder = target_folder / "backup"
        
        target_folder.mkdir(exist_ok=True)
        backup_folder.mkdir(exist_ok=True)
        
        # Find all WAV files in the subfolder
        wav_files = list(subfolder.glob("*.wav"))
        
        if not wav_files:
            print(f"  No WAV files found in {word_name}")
            continue
        
        print(f"  Found {len(wav_files)} WAV files")
        
        # Copy each WAV file to target folder
        for wav_file in wav_files:
            target_file = target_folder / wav_file.name
            print(f"  Copying: {wav_file.name}")
            
            try:
                shutil.copy2(wav_file, target_file)
                total_files_copied += 1
            except Exception as e:
                print(f"  Error copying {wav_file.name}: {e}")
                continue
    
    print(f"\nCopy operation completed!")
    print(f"Total files copied: {total_files_copied}")
    print(f"Data structure created in: {data_dir}")
    
    return True

def main():
    """Main function to run the copy operation."""
    print("=" * 50)
    print("Audio Recordings Copy Script")
    print("=" * 50)
    
    success = copy_recordings()
    
    if success:
        print("\n[SUCCESS] Copy operation completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run audio_split.py to split the recordings by silence")
        print("3. Each recording will be split into individual word segments")
    else:
        print("\n[ERROR] Copy operation failed!")
    
    return success

if __name__ == "__main__":
    main()

