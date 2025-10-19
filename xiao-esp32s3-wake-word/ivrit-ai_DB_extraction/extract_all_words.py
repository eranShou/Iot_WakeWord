#!/usr/bin/env python3
"""
Extract ALL word-level audio from ivrit-ai dataset
This script systematically extracts every occurrence of target words
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
import librosa
import soundfile as sf

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def strip_punctuation(word):
    """Strip punctuation from word for exact matching"""
    return word.strip().strip('!.,?;:')

def extract_word_audio_robust(audio_file, start_time, end_time, sample_rate=16000):
    """
    Extract audio segment with robust error handling and retry logic
    """
    temp_wav_path = None
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Convert .mka to .wav using ffmpeg with better error handling
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Try FFmpeg conversion with longer timeout
            result = subprocess.run([
                'ffmpeg', '-i', str(audio_file), 
                '-ar', str(sample_rate), '-ac', '1', 
                '-y', temp_wav_path
            ], capture_output=True, timeout=180)  # 3 minute timeout
            
            if result.returncode == 0:
                break  # Success
            else:
                if attempt < max_retries - 1:
                    print(f"FFmpeg attempt {attempt + 1} failed for {audio_file}, retrying...")
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"FFmpeg failed after {max_retries} attempts for {audio_file}")
                    return None
                    
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                print(f"FFmpeg timeout for {audio_file}, retrying...")
                time.sleep(2)
            else:
                print(f"FFmpeg timeout after {max_retries} attempts for {audio_file}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error on attempt {attempt + 1} for {audio_file}: {e}, retrying...")
                time.sleep(2)
            else:
                print(f"Failed after {max_retries} attempts for {audio_file}: {e}")
                return None
    
    try:
        # Load the converted audio
        audio, sr = librosa.load(temp_wav_path, sr=sample_rate, mono=True)
        
        # Extract the specific word segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        word_audio = audio[start_sample:end_sample]
        
        # Pad to exactly 1 second (16000 samples)
        if len(word_audio) < 16000:
            word_audio = librosa.util.pad_center(word_audio, size=16000)
        elif len(word_audio) > 16000:
            # If longer, take center portion
            start_center = (len(word_audio) - 16000) // 2
            word_audio = word_audio[start_center:start_center + 16000]
        
        return word_audio
        
    except Exception as e:
        print(f"Error processing audio segment: {e}")
        return None
    finally:
        # Clean up temporary file if it exists
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass

def find_all_word_occurrences(dataset_path, target_words):
    """
    Find ALL occurrences of target words in the dataset
    """
    print("Scanning dataset for ALL word occurrences...")
    print("=" * 60)
    
    all_occurrences = {word_key: [] for word_key in target_words.keys()}
    all_occurrences['unknown'] = []  # Add unknown category
    total_files_processed = 0
    files_with_matches = 0
    unknown_count = 0
    max_unknown = 2000  # Limit unknown words to 2000
    
    dataset_root = Path(dataset_path)
    
    # Get all transcript files first
    transcript_files = list(dataset_root.glob("*/transcript.aligned.json"))
    print(f"Found {len(transcript_files)} transcript files to process")
    
    for i, transcript_file in enumerate(transcript_files):
        total_files_processed += 1
        
        if total_files_processed % 100 == 0:
            print(f"Processed {total_files_processed}/{len(transcript_files)} files...")
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Process all segments
            segments = transcript_data.get('segments', [])
            file_has_matches = False
            
            for segment in segments:
                words = segment.get('words', [])
                
                for word_data in words:
                    word_text = word_data.get('word', '').strip()
                    word_clean = strip_punctuation(word_text)
                    
                    # Check if this word contains our target words (not exact match)
                    matched_target = None
                    for target_name, target_hebrew in target_words.items():
                        if target_hebrew in word_clean:
                            matched_target = target_name
                            break
                    
                    if matched_target:
                        # Found a target word match! Store the occurrence details
                        occurrence = {
                            'transcript_file': transcript_file,
                            'audio_file': transcript_file.parent / "audio.mka",
                            'word_text': word_text,
                            'word_clean': word_clean,
                            'start_time': word_data.get('start', 0),
                            'end_time': word_data.get('end', 0),
                            'folder_name': transcript_file.parent.name
                        }
                        all_occurrences[matched_target].append(occurrence)
                        file_has_matches = True
                    elif unknown_count < max_unknown and len(word_clean) > 2:
                        # Collect unknown words (not target words, longer than 2 chars)
                        occurrence = {
                            'transcript_file': transcript_file,
                            'audio_file': transcript_file.parent / "audio.mka",
                            'word_text': word_text,
                            'word_clean': word_clean,
                            'start_time': word_data.get('start', 0),
                            'end_time': word_data.get('end', 0),
                            'folder_name': transcript_file.parent.name
                        }
                        all_occurrences['unknown'].append(occurrence)
                        unknown_count += 1
                        file_has_matches = True
            
            if file_has_matches:
                files_with_matches += 1
                
        except Exception as e:
            print(f"Error processing {transcript_file}: {e}")
            continue
    
    print(f"\nScanning complete!")
    print(f"Files processed: {total_files_processed}")
    print(f"Files with matches: {files_with_matches}")
    for word_key, occurrences in all_occurrences.items():
        print(f"{word_key}: {len(occurrences)} occurrences found")
    
    print(f"\nUnknown words collected: {unknown_count}/{max_unknown}")
    
    return all_occurrences

def extract_all_occurrences(occurrences, output_dir, sample_rate=16000):
    """
    Extract audio for ALL occurrences found
    """
    print(f"\nExtracting audio for all occurrences...")
    print("=" * 60)
    
    total_extracted = 0
    total_failed = 0
    
    for word_key, word_occurrences in occurrences.items():
        if len(word_occurrences) == 0:
            continue
            
        print(f"\nProcessing {len(word_occurrences)} {word_key} occurrences...")
        
        for i, occurrence in enumerate(word_occurrences):
            try:
                # Extract audio for this occurrence
                word_audio = extract_word_audio_robust(
                    occurrence['audio_file'],
                    occurrence['start_time'],
                    occurrence['end_time'],
                    sample_rate
                )
                
                if word_audio is not None:
                    # Save the extracted word audio
                    output_filename = f"{word_key}_{i+1:03d}.wav"
                    output_path = output_dir / word_key / output_filename
                    
                    # Ensure parent directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(output_path), word_audio, sample_rate)
                    
                    total_extracted += 1
                    
                    if total_extracted % 50 == 0:
                        print(f"Extracted {total_extracted} words so far...")
                        
                else:
                    total_failed += 1
                    if total_failed <= 10:  # Only print first 10 failures
                        print(f"Failed to extract {word_key} from {occurrence['folder_name']}")
                    elif total_failed == 11:
                        print("... (suppressing further failure messages)")
                        
            except Exception as e:
                total_failed += 1
                if total_failed <= 10:
                    print(f"Error extracting {word_key} from {occurrence['folder_name']}: {e}")
    
    print(f"\nExtraction complete!")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    
    return total_extracted, total_failed

def main():
    """
    Main function to extract ALL word occurrences
    """
    print("Hebrew Complete Word-Level Audio Extraction")
    print("=" * 70)
    
    config = load_config()
    dataset_path = Path(config['dataset_path'])
    output_dir = Path(config['output_dir'])
    sample_rate = config['audio']['sample_rate']
    target_words = config['target_words']
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return False
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for word_key in target_words.keys():
        (output_dir / word_key).mkdir(parents=True, exist_ok=True)
    (output_dir / "unknown").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find ALL occurrences
    all_occurrences = find_all_word_occurrences(dataset_path, target_words)
    
    # Step 2: Extract audio for ALL occurrences
    total_extracted, total_failed = extract_all_occurrences(all_occurrences, output_dir, sample_rate)
    
    # Create summary
    summary = {
        "total_extracted": total_extracted,
        "total_failed": total_failed,
        "extraction_details": {word_key: len(occurrences) for word_key, occurrences in all_occurrences.items()},
        "sample_rate": sample_rate,
        "duration_seconds": 1.0,
        "extraction_method": "complete_word_level_timing"
    }
    
    with open(output_dir / "complete_extraction_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Complete extraction finished!")
    print(f"Check {output_dir} for all extracted word audio files")
    print(f"Summary saved to {output_dir}/complete_extraction_summary.json")
    
    return True

if __name__ == "__main__":
    main()
