"""
Download and process ivrit-ai/crowd-recital dataset
Extract 'shalom' and 'lehitraoot' recordings for training data
Collect all other recordings as 'unknown' for negative examples
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, login
import librosa
import soundfile as sf

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def load_credentials():
    """Load HuggingFace credentials from secret.json"""
    try:
        with open('secret.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: secret.json file not found. Please create it with your HuggingFace credentials.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in secret.json")
        return None

def authenticate_huggingface():
    """Authenticate with HuggingFace using credentials from secret.json"""
    credentials = load_credentials()
    if not credentials:
        return False
    
    try:
        hf_creds = credentials.get('huggingface', {})
        token = hf_creds.get('token')
        
        if not token:
            print("Error: HuggingFace token not found in secret.json")
            print("Please get your token from: https://huggingface.co/settings/tokens")
            print("Add it to secret.json as: {\"huggingface\": {\"token\": \"your_token_here\"}}")
            return False
        
        print("Authenticating with HuggingFace...")
        login(token=token)
        print("✓ Authentication successful!")
        return True
        
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False

def download_dataset():
    """
    Download ivrit-ai/crowd-recital dataset from HuggingFace
    """
    print("Downloading ivrit-ai/crowd-recital dataset...")
    print("=" * 50)
    
    try:
        # Download dataset with timeout settings for stability
        dataset_path = snapshot_download(
            repo_id='ivrit-ai/crowd-recital',
            repo_type='dataset',
            etag_timeout=60,
            max_workers=1
        )
        
        print(f"✓ Dataset downloaded to: {dataset_path}")
        return dataset_path
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("Try increasing timeout or reducing workers if you encounter issues")
        return None

def find_all_recordings(dataset_path, target_words):
    """
    Find all recordings in the dataset and categorize them
    """
    print(f"\nSearching for all recordings in dataset...")
    print(f"Target words: {target_words}")
    print("=" * 50)
    
    found_recordings = {word: [] for word in target_words}
    found_recordings['unknown'] = []
    
    # Search through dataset structure
    dataset_root = Path(dataset_path)
    
    # Hebrew target words
    hebrew_words = {
        'shalom': ['שלום', 'שלום!', 'שלום.', 'שלום,'],
        'lehitraoot': ['להתראות', 'להתראות!', 'להתראות.', 'להתראות,', 'להתראות?']
    }
    
    # Search through all folders in the dataset
    for folder in dataset_root.iterdir():
        if folder.is_dir():
            # Check for transcript files
            transcript_file = folder / "transcript.aligned.json"
            if transcript_file.exists():
                try:
                    # Load transcript
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                    
                    # Get the text content
                    text = transcript_data.get('text', '')
                    
                    # Check if text contains our target words
                    matched_target = False
                    for word, hebrew_variants in hebrew_words.items():
                        for variant in hebrew_variants:
                            if variant in text:
                                # Find the corresponding audio file
                                audio_file = folder / "audio.mka"
                                if audio_file.exists():
                                    found_recordings[word].append(audio_file)
                                    print(f"Found {word} ({variant}): {audio_file}")
                                    matched_target = True
                                    break
                        if matched_target:
                            break
                    
                    # If no target word matched, add to unknown
                    if not matched_target:
                        audio_file = folder / "audio.mka"
                        if audio_file.exists():
                            found_recordings['unknown'].append(audio_file)
                            if len(found_recordings['unknown']) <= 10:  # Only print first 10 unknown
                                print(f"Found unknown: {audio_file}")
                            elif len(found_recordings['unknown']) == 11:
                                print("... (and more unknown recordings)")
                
                except Exception as e:
                    print(f"Error processing {transcript_file}: {e}")
                    continue
    
    # Print summary
    for word, recordings in found_recordings.items():
        print(f"{word}: {len(recordings)} recordings found")
    
    return found_recordings

def process_and_organize_recordings(found_recordings, output_base_dir):
    """
    Process found recordings and organize into target directories
    """
    print(f"\nProcessing and organizing recordings...")
    print("=" * 50)
    
    config = load_config()
    sample_rate = config['audio']['sample_rate']
    duration_seconds = config['audio']['duration_seconds']
    num_samples = config['audio']['num_samples']
    
    processed_count = 0
    
    for word, recordings in found_recordings.items():
        if not recordings:
            print(f"No recordings found for {word}")
            continue
        
        # Create output directory
        output_dir = Path(output_base_dir) / "ivrit-ai" / word
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {word} recordings...")
        
        for i, recording_path in enumerate(recordings):
            try:
                # Load audio file (handle .mka files)
                if recording_path.suffix.lower() == '.mka':
                    # Use ffmpeg to convert .mka to temporary .wav
                    import subprocess
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        temp_wav_path = temp_wav.name
                    
                    # Convert .mka to .wav using ffmpeg
                    subprocess.run([
                        'ffmpeg', '-i', str(recording_path), 
                        '-ar', str(sample_rate), '-ac', '1', 
                        '-y', temp_wav_path
                    ], capture_output=True, check=True)
                    
                    # Load the converted audio
                    audio, sr = librosa.load(temp_wav_path, sr=sample_rate, mono=True)
                    
                    # Clean up temporary file
                    os.unlink(temp_wav_path)
                else:
                    # Load audio file directly
                    audio, sr = librosa.load(str(recording_path), sr=sample_rate, mono=True)
                
                # Skip if too short
                if len(audio) < num_samples // 2:  # At least 0.5 seconds
                    continue
                
                # If longer than target duration, take center portion
                if len(audio) > num_samples:
                    start = (len(audio) - num_samples) // 2
                    audio = audio[start:start + num_samples]
                else:
                    # Pad with zeros if shorter
                    audio = librosa.util.pad_center(audio, num_samples)
                
                # Save processed audio
                output_filename = f"{word}_{i+1:03d}.wav"
                output_path = output_dir / output_filename
                
                sf.write(str(output_path), audio, sample_rate)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} recordings...")
                
            except Exception as e:
                print(f"Error processing {recording_path}: {e}")
                continue
    
    print(f"\n✓ Processing complete!")
    print(f"Total processed recordings: {processed_count}")
    
    # Print final directory structure
    print(f"\nOutput directory structure:")
    all_words = list(found_recordings.keys())
    for word in all_words:
        word_dir = Path(output_base_dir) / "ivrit-ai" / word
        if word_dir.exists():
            wav_files = list(word_dir.glob('*.wav'))
            print(f"  {word}: {len(wav_files)} files in {word_dir}")

def main():
    """
    Main function to download and process ivrit-ai dataset
    """
    print("Hebrew Dataset Download and Processing")
    print("=" * 60)
    
    # Target words to extract
    target_words = ['shalom', 'lehitraoot']
    
    # Output directory
    output_base_dir = "./data"
    
    # Step 0: Authenticate with HuggingFace
    if not authenticate_huggingface():
        print("Authentication failed. Cannot proceed with dataset download.")
        return False
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("Failed to download dataset")
        return False
    
    # Step 2: Find all recordings
    found_recordings = find_all_recordings(dataset_path, target_words)
    
    # Step 3: Process and organize
    process_and_organize_recordings(found_recordings, output_base_dir)
    
    print(f"\n✓ Dataset processing completed successfully!")
    print(f"Check ./data/ivrit-ai/ for organized recordings")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ivrit-ai dataset processing completed!")
    else:
        print("\n❌ Dataset processing failed!")