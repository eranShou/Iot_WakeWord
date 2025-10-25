from IvritAiDataFetch import IvritAiDataFetcher
import os
import gzip
import json
import random
from pydub import AudioSegment
import re
import glob

EXCLUDE_WORDS = {"שלום", "להתראות"}
SNIPPETS_PER_FILE = 10

def extract_random_word_snippets(transcript_path, audio_dir, output_base, exclude_words, n=40):
    base_name = os.path.basename(transcript_path).replace("_transcript.json.gz", "")
    # Find corresponding audio file
    audio_file = None
    for ext in [".mp3", ".m4a", ".wav", ".flac"]:
        candidate = os.path.join(audio_dir, base_name + ext)
        if os.path.exists(candidate):
            audio_file = candidate
            break
    if not audio_file:
        print(f"No audio file found for {base_name}")
        return
    # Load transcript
    with gzip.open(transcript_path, "rt", encoding="utf-8") as f:
        try:
            transcript_json = json.load(f)
        except Exception as e:
            print(f"Failed to load {transcript_path}: {e}")
            return
    # Load audio
    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        print(f"Failed to load audio {audio_file}: {e}")
        return
    # Collect all eligible word occurrences
    segments = transcript_json.get("segments", [])
    words_starting_with_shin = []
    words_ending_with_um = []

    for seg in segments:
        words_info = seg.get("words", [])
        for word_info in words_info:
            word_text = word_info.get("word", "").strip().lower()
            word_start = word_info.get("start", None)
            word_end = word_info.get("end", None)
            if word_start is None or word_end is None:
                continue
            if word_text not in exclude_words:
                if word_text.startswith('ש'):
                    words_starting_with_shin.append((word_text, word_start, word_end))
                elif word_text.endswith('ום'):
                    words_ending_with_um.append((word_text, word_start, word_end))

    # Sample 5 from each array and merge
    sampled_words = random.sample(words_starting_with_shin, min(5, len(words_starting_with_shin))) + \
                    random.sample(words_ending_with_um, min(5, len(words_ending_with_um)))

    if not sampled_words:
        print(f"No eligible words found in {transcript_path}")
        return
    
    # Randomly select up to n words
    chosen = sampled_words
    # Save audio snippets
    for idx, (word_text, word_start, word_end) in enumerate(chosen):
        buffer_ms = 100
        word_start_ms = int(float(word_start) * 1000)
        word_end_ms = int(float(word_end) * 1000)
        clip_start = max(0, word_start_ms - buffer_ms)
        clip_end = min(len(audio), word_end_ms + buffer_ms)
        if clip_end > clip_start:
            clip = audio[clip_start:clip_end]
            # Sanitize the word_text to make it a valid filename, allowing Hebrew characters
            sanitized_word_text = re.sub(r'[^a-zA-Z0-9_\-\u0590-\u05FF]', '_', word_text)
            out_name = f"{sanitized_word_text}_{word_start_ms}_{word_end_ms}_buffer.wav"
            out_path = os.path.join(output_base, out_name)
            clip.export(out_path, format="wav")
            print(f"Saved {out_path}")

if __name__ == "__main__":
    # List the first 50 files in the audio directory
    audio_dir = "D:\\ivirit-ai-data\\ivrit-ai dataset\\audio_v2_downloaded"
    transcript_dir = "D:\\ivirit-ai-data\\ivrit-ai dataset\\transcripts_v2_downloaded"
    output_dir = "D:\\ivirit-ai-data\\ivrit-ai dataset\\word_clips"

    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3"))[:50]

    # Strip out the .mp3 extension to get the base names
    base_names = [os.path.splitext(os.path.basename(file))[0] for file in audio_files]

    # Run random word snippet extraction on each base name
    for base_name in base_names:
        transcript_path = os.path.join(transcript_dir, f"{base_name}_transcript.json.gz")
        extract_random_word_snippets(
            transcript_path=transcript_path,
            audio_dir=audio_dir,
            output_base=output_dir,
            exclude_words=EXCLUDE_WORDS,
            n=SNIPPETS_PER_FILE
        )
