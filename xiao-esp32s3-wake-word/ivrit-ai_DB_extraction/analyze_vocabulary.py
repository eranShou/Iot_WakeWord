#!/usr/bin/env python3
"""
Analyze vocabulary of the entire ivrit-ai dataset
This script scans all transcript files and creates a comprehensive word frequency analysis
"""

import os
import json
import time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def strip_punctuation(word):
    """Strip punctuation from word for clean analysis"""
    return word.strip().strip('!.,?;:()[]{}"\'`~@#$%^&*+=|\\/<>')

def process_single_file(transcript_file):
    """
    Process a single transcript file and return word counts
    """
    file_word_counts = Counter()
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Process all segments
        segments = transcript_data.get('segments', [])
        
        for segment in segments:
            words = segment.get('words', [])
            
            for word_data in words:
                word_text = word_data.get('word', '').strip()
                word_clean = strip_punctuation(word_text)
                
                # Only count meaningful words (longer than 1 character, not just punctuation)
                if len(word_clean) > 1 and word_clean.isalpha():
                    file_word_counts[word_clean] += 1
                    
    except Exception as e:
        print(f"Error processing {transcript_file}: {e}")
        return Counter()
    
    return file_word_counts

def analyze_vocabulary_parallel(dataset_path, max_workers=None):
    """
    Analyze vocabulary of the entire dataset using parallel processing
    """
    print("Analyzing vocabulary of the entire ivrit-ai dataset...")
    print("=" * 70)
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    print(f"Using {max_workers} parallel workers")
    
    dataset_root = Path(dataset_path)
    transcript_files = list(dataset_root.glob("*/transcript.aligned.json"))
    print(f"Found {len(transcript_files)} transcript files to analyze")
    
    # Global word counter with thread safety
    global_word_counts = Counter()
    word_counts_lock = threading.Lock()
    
    total_files_processed = 0
    total_words_processed = 0
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_file, transcript_file): transcript_file
            for transcript_file in transcript_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            transcript_file = future_to_file[future]
            total_files_processed += 1
            
            if total_files_processed % 200 == 0:
                print(f"Processed {total_files_processed}/{len(transcript_files)} files...")
            
            try:
                file_word_counts = future.result()
                
                # Merge results thread-safely
                with word_counts_lock:
                    for word, count in file_word_counts.items():
                        global_word_counts[word] += count
                        total_words_processed += count
                        
            except Exception as e:
                print(f"Error processing {transcript_file}: {e}")
                continue
    
    print(f"\nAnalysis complete!")
    print(f"Files processed: {total_files_processed}")
    print(f"Total words processed: {total_words_processed}")
    print(f"Unique words found: {len(global_word_counts)}")
    
    return global_word_counts

def create_vocabulary_analysis(word_counts, output_dir):
    """
    Create comprehensive vocabulary analysis and save to JSON
    """
    print(f"\nCreating vocabulary analysis...")
    print("=" * 50)
    
    # Sort words by frequency (most common first)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create analysis data
    analysis_data = {
        "metadata": {
            "total_unique_words": len(word_counts),
            "total_word_occurrences": sum(word_counts.values()),
            "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "ivrit-ai/crowd-recital"
        },
        "word_frequencies": dict(sorted_words),
        "top_words": {
            "most_common": sorted_words[:100],  # Top 100 most common words
            "least_common": sorted_words[-100:] if len(sorted_words) > 100 else sorted_words  # Bottom 100
        },
        "statistics": {
            "words_appearing_once": len([word for word, count in word_counts.items() if count == 1]),
            "words_appearing_10_plus": len([word for word, count in word_counts.items() if count >= 10]),
            "words_appearing_100_plus": len([word for word, count in word_counts.items() if count >= 100]),
            "words_appearing_1000_plus": len([word for word, count in word_counts.items() if count >= 1000]),
            "average_frequency": sum(word_counts.values()) / len(word_counts) if word_counts else 0
        }
    }
    
    # Save comprehensive analysis
    output_file = output_dir / "vocabulary_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    # Save simple word count dictionary
    simple_output_file = output_dir / "word_counts.json"
    with open(simple_output_file, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted_words), f, indent=2, ensure_ascii=False)
    
    print(f"Vocabulary analysis saved to: {output_file}")
    print(f"Simple word counts saved to: {simple_output_file}")
    
    # Print summary statistics
    print(f"\nVocabulary Analysis Summary:")
    print(f"  - Total unique words: {len(word_counts)}")
    print(f"  - Total word occurrences: {sum(word_counts.values())}")
    print(f"  - Most common word: '{sorted_words[0][0]}' ({sorted_words[0][1]} times)")
    print(f"  - Words appearing once: {analysis_data['statistics']['words_appearing_once']}")
    print(f"  - Words appearing 10+ times: {analysis_data['statistics']['words_appearing_10_plus']}")
    print(f"  - Words appearing 100+ times: {analysis_data['statistics']['words_appearing_100_plus']}")
    print(f"  - Words appearing 1000+ times: {analysis_data['statistics']['words_appearing_1000_plus']}")
    
    # Print top 20 most common words
    print(f"\nTop 20 Most Common Words:")
    for i, (word, count) in enumerate(sorted_words[:20], 1):
        print(f"  {i:2d}. {word:<20} ({count:>6} times)")
    
    return analysis_data

def main():
    """
    Main function to analyze vocabulary of the entire ivrit-ai dataset
    """
    print("Hebrew Vocabulary Analysis - ivrit-ai Dataset")
    print("=" * 70)
    
    # Get CPU info
    total_cores = cpu_count()
    max_workers = max(1, total_cores - 1)
    print(f"Detected {total_cores} CPU cores, using {max_workers} workers")
    
    config = load_config()
    dataset_path = Path(config['dataset_path'])
    # Save output files in the same directory as the script
    output_dir = Path(__file__).parent
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze vocabulary (parallel)
    start_time = time.time()
    word_counts = analyze_vocabulary_parallel(dataset_path, max_workers)
    analysis_time = time.time() - start_time
    print(f"Vocabulary analysis completed in {analysis_time:.2f} seconds")
    
    # Step 2: Create analysis files
    start_time = time.time()
    analysis_data = create_vocabulary_analysis(word_counts, output_dir)
    creation_time = time.time() - start_time
    print(f"Analysis files created in {creation_time:.2f} seconds")
    
    print(f"\n✓ Vocabulary analysis completed!")
    print(f"Performance: {max_workers}/{total_cores} cores used")
    print(f"Total time: {analysis_time + creation_time:.2f} seconds")
    print(f"Check {output_dir} for vocabulary analysis files")
    
    return True

if __name__ == "__main__":
    main()
