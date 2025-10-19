#!/usr/bin/env python3
"""
Extract ALL word-level audio from ivrit-ai dataset with optimized parallel processing
This script uses all CPU cores (minus 1) and limits unknown words intelligently
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import random

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_vocabulary_counts():
    """Load word frequencies from vocabulary_analysis.json"""
    try:
        with open('vocabulary_analysis.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('word_frequencies', {})
    except FileNotFoundError:
        print("Warning: vocabulary_analysis.json not found. Will extract all occurrences.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading vocabulary_analysis.json: {e}. Will extract all occurrences.")
        return {}

def strip_punctuation(word):
    """Strip punctuation from word for exact matching"""
    return word.strip().strip('!.,?;:()[]{}"\'`~@#$%^&*+=|\\/<>')

def extract_word_audio_robust(audio_file, start_time, end_time, sample_rate=16000):
    """
    Extract audio segment with robust error handling and retry logic
    """
    temp_wav_path = None
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            result = subprocess.run([
                'ffmpeg', '-i', str(audio_file), 
                '-ar', str(sample_rate), '-ac', '1', 
                '-y', temp_wav_path
            ], capture_output=True, timeout=60)
            
            if result.returncode == 0:
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None
                    
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    
    try:
        audio, sr = librosa.load(temp_wav_path, sr=sample_rate, mono=True)
        
        # Add 50ms margin to account for timing errors
        margin = 0.05  # 50ms
        adjusted_start_time = max(0, start_time - margin)
        adjusted_end_time = end_time + margin
        
        start_sample = int(adjusted_start_time * sample_rate)
        end_sample = int(adjusted_end_time * sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        word_audio = audio[start_sample:end_sample]
        
        # Ensure exactly 1 second (16000 samples)
        if len(word_audio) < 16000:
            # Pad with silence, centered
            word_audio = librosa.util.pad_center(word_audio, size=16000)
        elif len(word_audio) > 16000:
            # Center-crop to 1 second
            start_center = (len(word_audio) - 16000) // 2
            word_audio = word_audio[start_center:start_center + 16000]
        
        return word_audio
        
    except Exception as e:
        return None
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass

def process_single_file(transcript_file, target_words, max_unknown, unknown_lock, unknown_count):
    """
    Process a single transcript file and return all occurrences found
    """
    file_occurrences = {word_key: [] for word_key in target_words.keys()}
    file_occurrences['unknown'] = []
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        segments = transcript_data.get('segments', [])
        
        for segment in segments:
            words = segment.get('words', [])
            
            for word_data in words:
                word_text = word_data.get('word', '').strip()
                word_clean = strip_punctuation(word_text)
                
                # Check if this word exactly matches our target words
                matched_target = None
                for target_name, target_hebrew in target_words.items():
                    if word_clean == target_hebrew:
                        matched_target = target_name
                        break
                
                if matched_target:
                    occurrence = {
                        'transcript_file': transcript_file,
                        'audio_file': transcript_file.parent / "audio.mka",
                        'word_text': word_text,
                        'word_clean': word_clean,
                        'start_time': word_data.get('start', 0),
                        'end_time': word_data.get('end', 0),
                        'folder_name': transcript_file.parent.name
                    }
                    file_occurrences[matched_target].append(occurrence)
                else:
                    # Check if we can add to unknown (with better filtering)
                    with unknown_lock:
                        if (unknown_count[0] < max_unknown and 
                            len(word_clean) > 3 and  # Longer words only
                            len(word_clean) < 20 and  # Not too long
                            not word_clean.isdigit() and  # Not just numbers
                            word_clean.isalpha()):  # Only alphabetic characters
                            
                            occurrence = {
                                'transcript_file': transcript_file,
                                'audio_file': transcript_file.parent / "audio.mka",
                                'word_text': word_text,
                                'word_clean': word_clean,
                                'start_time': word_data.get('start', 0),
                                'end_time': word_data.get('end', 0),
                                'folder_name': transcript_file.parent.name
                            }
                            file_occurrences['unknown'].append(occurrence)
                            unknown_count[0] += 1
                            
    except Exception as e:
        return file_occurrences
    
    return file_occurrences

def check_existing_extractions(output_dir, target_words, vocabulary_counts):
    """
    Check which target words are already fully extracted based on vocabulary counts
    Returns dict of {word_key: {'existing': count, 'expected': count, 'skip': bool}}
    """
    extraction_status = {}
    
    for word_key, target_hebrew in target_words.items():
        word_dir = output_dir / word_key
        if word_dir.exists():
            existing_count = len(list(word_dir.glob("*.wav")))
        else:
            existing_count = 0
        
        expected_count = vocabulary_counts.get(target_hebrew, 0)
        
        # Skip if we have enough files (within ±1 tolerance)
        skip = existing_count >= max(1, expected_count - 1) if expected_count > 0 else True
        
        extraction_status[word_key] = {
            'existing': existing_count,
            'expected': expected_count,
            'skip': skip
        }
    
    return extraction_status

def find_all_word_occurrences_parallel(dataset_path, target_words, max_workers=None, extraction_status=None):
    """
    Find ALL occurrences of target words in the dataset using parallel processing
    """
    print("Scanning dataset for ALL word occurrences (PARALLEL)...")
    print("=" * 60)
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    if extraction_status is None:
        extraction_status = {word_key: {'existing': 0, 'expected': 0, 'skip': False} for word_key in target_words.keys()}
    
    # Track which words to actually scan for (skip words that already have enough files)
    words_to_scan = {}
    for word_key, target_hebrew in target_words.items():
        status = extraction_status.get(word_key, {'existing': 0, 'expected': 0, 'skip': False})
        if status['skip']:
            print(f"Skipping {word_key}: already have {status['existing']} files (vocabulary shows {status['expected']} occurrences)")
        else:
            print(f"Will process {word_key}: have {status['existing']} files, need {status['expected']} (vocabulary shows {status['expected']} occurrences)")
            words_to_scan[word_key] = target_hebrew
    
    if not words_to_scan:
        print("All words already have sufficient files. No scanning needed.")
        return {word_key: [] for word_key in target_words.keys()}
    
    print(f"Using {max_workers} parallel workers")
    
    all_occurrences = {word_key: [] for word_key in target_words.keys()}
    all_occurrences['unknown'] = []
    
    dataset_root = Path(dataset_path)
    transcript_files = list(dataset_root.glob("*/transcript.aligned.json"))
    print(f"Found {len(transcript_files)} transcript files to process")
    
    unknown_count = [0]
    unknown_lock = threading.Lock()
    max_unknown = 2000
    
    total_files_processed = 0
    files_with_matches = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, transcript_file, words_to_scan, max_unknown, unknown_lock, unknown_count): transcript_file
            for transcript_file in transcript_files
        }
        
        for future in as_completed(future_to_file):
            transcript_file = future_to_file[future]
            total_files_processed += 1
            
            if total_files_processed % 200 == 0:
                print(f"Processed {total_files_processed}/{len(transcript_files)} files...")
            
            try:
                file_occurrences = future.result()
                
                file_has_matches = False
                for word_key, occurrences in file_occurrences.items():
                    if len(occurrences) > 0:
                        all_occurrences[word_key].extend(occurrences)
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
    
    # Check which words need extraction based on vocabulary counts
    print(f"\nChecking which words need extraction...")
    for word_key, occurrences in all_occurrences.items():
        if word_key == 'unknown':
            continue
        scanned_count = len(occurrences)
        status = extraction_status.get(word_key, {'existing': 0, 'expected': 0, 'skip': False})
        expected_count = status['expected']
        
        if status['skip']:
            print(f"Skipping {word_key}: already have {status['existing']} files (vocabulary shows {expected_count} occurrences)")
            all_occurrences[word_key] = []  # Clear to skip extraction
        elif scanned_count > 0:
            print(f"Will extract {word_key}: found {scanned_count} occurrences (vocabulary shows {expected_count})")
        else:
            print(f"No occurrences found for {word_key} (vocabulary shows {expected_count})")
    
    print(f"\nUnknown words collected: {unknown_count[0]}/{max_unknown}")
    
    return all_occurrences

def extract_audio_parallel(occurrences, output_dir, sample_rate=16000, max_workers=None, extraction_status=None):
    """
    Extract audio for ALL occurrences using parallel processing
    """
    print(f"\nExtracting audio for all occurrences (PARALLEL)...")
    print("=" * 60)
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    # Delete existing files for words that need re-extraction
    if extraction_status:
        for word_key, word_occurrences in occurrences.items():
            if len(word_occurrences) > 0 and word_key != 'unknown':
                status = extraction_status.get(word_key, {'existing': 0, 'expected': 0, 'skip': False})
                if not status['skip'] and status['existing'] > 0:
                    word_dir = output_dir / word_key
                    if word_dir.exists():
                        print(f"Deleting {status['existing']} existing files for {word_key} before re-extraction...")
                        for existing_file in word_dir.glob("*.wav"):
                            existing_file.unlink()
    
    print(f"Using {max_workers} parallel workers for audio extraction")
    
    total_extracted = 0
    total_failed = 0
    extraction_lock = threading.Lock()
    
    def extract_single_occurrence(occurrence_data):
        nonlocal total_extracted, total_failed
        word_key, i, occurrence = occurrence_data
        
        try:
            word_audio = extract_word_audio_robust(
                occurrence['audio_file'],
                occurrence['start_time'],
                occurrence['end_time'],
                sample_rate
            )
            
            if word_audio is not None:
                output_filename = f"{word_key}_{i+1:03d}.wav"
                output_path = output_dir / word_key / output_filename
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), word_audio, sample_rate)
                
                with extraction_lock:
                    total_extracted += 1
                    if total_extracted % 100 == 0:
                        print(f"Extracted {total_extracted} words so far...")
                
                return True
            else:
                with extraction_lock:
                    total_failed += 1
                    if total_failed <= 10:
                        print(f"Failed to extract {word_key} from {occurrence['folder_name']}")
                    elif total_failed == 11:
                        print("... (suppressing further failure messages)")
                
                return False
                
        except Exception as e:
            with extraction_lock:
                total_failed += 1
                if total_failed <= 10:
                    print(f"Error extracting {word_key} from {occurrence['folder_name']}: {e}")
            
            return False
    
    all_tasks = []
    for word_key, word_occurrences in occurrences.items():
        if len(word_occurrences) == 0:
            continue
        
        print(f"Preparing {len(word_occurrences)} {word_key} occurrences for parallel extraction...")
        
        for i, occurrence in enumerate(word_occurrences):
            all_tasks.append((word_key, i, occurrence))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(extract_single_occurrence, task): task
            for task in all_tasks
        }
        
        for future in as_completed(future_to_task):
            try:
                future.result()
            except Exception as e:
                print(f"Unexpected error in extraction: {e}")
    
    print(f"\nExtraction complete!")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    
    return total_extracted, total_failed

def main():
    """
    Main function to extract ALL word occurrences with optimized parallel processing
    """
    print("Hebrew Complete Word-Level Audio Extraction (OPTIMIZED)")
    print("=" * 70)
    
    total_cores = cpu_count()
    max_workers = max(1, total_cores - 1)
    print(f"Detected {total_cores} CPU cores, using {max_workers} workers")
    
    config = load_config()
    dataset_path = Path(config['dataset_path'])
    output_dir = Path(config['output_dir'])
    sample_rate = config['audio']['sample_rate']
    target_words = config['target_words']
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return False
    
    # Load vocabulary counts from vocabulary_analysis.json
    print("\nLoading vocabulary counts...")
    vocabulary_counts = load_vocabulary_counts()
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for word_key in target_words.keys():
        (output_dir / word_key).mkdir(parents=True, exist_ok=True)
    (output_dir / "unknown").mkdir(parents=True, exist_ok=True)
    
    # Check existing extractions against vocabulary counts
    print("\nChecking for existing extractions against vocabulary counts...")
    extraction_status = check_existing_extractions(output_dir, target_words, vocabulary_counts)
    
    # Step 1: Find ALL occurrences (parallel) - pass extraction status
    start_time = time.time()
    all_occurrences = find_all_word_occurrences_parallel(dataset_path, target_words, max_workers, extraction_status)
    scan_time = time.time() - start_time
    print(f"Scanning completed in {scan_time:.2f} seconds")
    
    # Step 2: Extract audio for ALL occurrences (parallel)
    start_time = time.time()
    total_extracted, total_failed = extract_audio_parallel(all_occurrences, output_dir, sample_rate, max_workers, extraction_status)
    extraction_time = time.time() - start_time
    print(f"Audio extraction completed in {extraction_time:.2f} seconds")
    
    # Create summary
    summary = {
        "total_extracted": total_extracted,
        "total_failed": total_failed,
        "extraction_details": {word_key: len(occurrences) for word_key, occurrences in all_occurrences.items()},
        "vocabulary_counts": {word_key: vocabulary_counts.get(target_words[word_key], 0) for word_key in target_words.keys()},
        "extraction_status": extraction_status,
        "sample_rate": sample_rate,
        "duration_seconds": 1.0,
        "extraction_method": "optimized_parallel_word_level_timing_with_vocabulary_skip",
        "performance": {
            "cpu_cores_used": max_workers,
            "total_cpu_cores": total_cores,
            "scan_time_seconds": scan_time,
            "extraction_time_seconds": extraction_time,
            "total_time_seconds": scan_time + extraction_time
        }
    }
    
    with open(output_dir / "optimized_extraction_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Optimized parallel extraction finished!")
    print(f"Performance: {max_workers}/{total_cores} cores used")
    print(f"Total time: {scan_time + extraction_time:.2f} seconds")
    print(f"Check {output_dir} for all extracted word audio files")
    print(f"Summary saved to {output_dir}/optimized_extraction_summary.json")
    
    return True

if __name__ == "__main__":
    main()
