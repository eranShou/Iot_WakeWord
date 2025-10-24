from IvritAiDataFetch import IvritAiDataFetcher
import os
import gzip
import json
import random
from pydub import AudioSegment
import re

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
    word_occurrences = []
    for seg in segments:
        words_info = seg.get("words", [])
        for word_info in words_info:
            word_text = word_info.get("word", "").strip().lower()
            word_start = word_info.get("start", None)
            word_end = word_info.get("end", None)
            if word_start is None or word_end is None:
                continue
            if word_text not in exclude_words:
                word_occurrences.append((word_text, word_start, word_end))
    if not word_occurrences:
        print(f"No eligible words found in {transcript_path}")
        return
    # Randomly select up to n words
    chosen = random.sample(word_occurrences, min(n, len(word_occurrences)))
    # Save audio snippets
    for idx, (word_text, word_start, word_end) in enumerate(chosen):
        buffer_ms = 50
        word_start_ms = int(float(word_start) * 1000)
        word_end_ms = int(float(word_end) * 1000)
        clip_start = max(0, word_start_ms - buffer_ms)
        clip_end = min(len(audio), word_end_ms + buffer_ms)
        if clip_end > clip_start:
            clip = audio[clip_start:clip_end]
            # Sanitize the word_text to make it a valid filename
            sanitized_word_text = re.sub(r'[^a-zA-Z0-9_\-]', '_', word_text)
            out_name = f"{base_name}_{sanitized_word_text}_{word_start_ms}_{word_end_ms}_buffer.wav"
            out_path = os.path.join(output_base, out_name)
            clip.export(out_path, format="wav")
            print(f"Saved {out_path}")

if __name__ == "__main__":
    fetcher = IvritAiDataFetcher()
    folders = fetcher.list_transcripts_root_folders()
    print(f"Found folders: {folders}")
    for folder in folders[:20]:
        print(f"Downloading first transcript in folder: {folder}")
        fetcher.download_episodes(path=folder, max_per_day=1)
        # After download, find the transcript file and audio, and extract random word snippets
        transcripts_dir = fetcher.OUTPUT_TRANSCRIPTS_DIR
        audio_dir = fetcher.OUTPUT_AUDIO_DIR
        output_base = os.path.join(fetcher.BASE_DIR, "random_word_snippets")
        print(output_base)
        os.makedirs(output_base, exist_ok=True)
        # Find the transcript file for this folder
        transcript_files = [file for file in os.listdir(transcripts_dir) if file.endswith("_transcript.json.gz")]
        selected_files = random.sample(transcript_files, min(40, len(transcript_files)))

        for file in selected_files:
            transcript_path = os.path.join(transcripts_dir, file)
            extract_random_word_snippets(transcript_path, audio_dir, output_base, EXCLUDE_WORDS, n=SNIPPETS_PER_FILE)