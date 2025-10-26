import os
import requests
from huggingface_hub import hf_hub_download, HfApi, Repository
import gzip
import json
from pydub import AudioSegment
import sys
import random
import re



sys.path.append(os.path.dirname(__file__))
from secrets import HF_TOKEN


class IvritAiDataFetcher:
    def __init__(self, hf_token=HF_TOKEN, base_dir=os.path.dirname(os.path.abspath(__file__)), buffer_ms=50):
        self.HF_TOKEN = hf_token
        self.DATASET_ID_AUDIO = "ivrit-ai/audio-v2"
        self.DATASET_ID_TRANSCRIPTS = "ivrit-ai/audio-v2-transcripts"
        self.BASE_DIR = base_dir
        self.OUTPUT_AUDIO_DIR = os.path.join(self.BASE_DIR, "audio_v2_downloaded")
        self.OUTPUT_TRANSCRIPTS_DIR = os.path.join(self.BASE_DIR, "transcripts_v2_downloaded")
        self.OUTPUT_WORD_CLIPS_DIR = os.path.join(self.BASE_DIR, "word_clips")
        self.BUFFER_MS = buffer_ms
        self.headers = {"Authorization": f"Bearer {self.HF_TOKEN}"}
        os.makedirs(self.OUTPUT_AUDIO_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_TRANSCRIPTS_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_WORD_CLIPS_DIR, exist_ok=True)
        self.api = HfApi()
    
    def list_transcripts_root_folders(self):
        """
        List all top-level folders in the HuggingFace transcripts repository (not recursive).
        Returns a list of folder names.
        """
        items = self.api.list_repo_tree(
            self.DATASET_ID_TRANSCRIPTS,
            recursive=False,
            repo_type="dataset",
            token=self.HF_TOKEN
        )

        folder_names = [item.path for item in items if item.__class__.__name__ == 'RepoFolder']
        print("Folder names:", folder_names)
        return folder_names
    
    def _download_file(self, dataset_id, file_path, local_path):
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{file_path}"
        resp = requests.get(url, headers=self.headers, stream=True)
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file_path} to {local_path}")
        else:
            print(f"Failed downloading {file_path}, status code {resp.status_code}")

    def _process_episode(self, episode_path):
        audio_extensions = [".mp3", ".m4a", ".wav", ".flac"]
        transcript_file = episode_path.rstrip("/\\") + "/full_transcript.json.gz"
        local_transcript = os.path.join(self.OUTPUT_TRANSCRIPTS_DIR, episode_path.replace("/", "_") + "_transcript.json.gz")

        # Check if the transcript file already exists locally
        if os.path.exists(local_transcript):
            print(f"Transcript already downloaded: {local_transcript}")
        else:
            try:
                self._download_file(self.DATASET_ID_TRANSCRIPTS, transcript_file, local_transcript)
            except Exception as e:
                print(f"Error downloading transcript for {episode_path}: {e}")

        for ext in audio_extensions:
            audio_file = episode_path + ext
            local_audio = os.path.join(self.OUTPUT_AUDIO_DIR, episode_path.replace("/", "_") + ext)

            # Check if the audio file already exists locally
            if os.path.exists(local_audio):
                print(f"Audio already downloaded: {local_audio}")
                break

            try:
                self._download_file(self.DATASET_ID_AUDIO, audio_file, local_audio)
                break
            except Exception as e:
                print(f"Audio with ext {ext} didn't work for {episode_path}: {e}")

    def download_episodes(self, path, episodes_per_dir):
        episode_dirs = self.api.list_repo_tree(self.DATASET_ID_TRANSCRIPTS, recursive=True, path_in_repo=path, repo_type="dataset", token=self.HF_TOKEN)
        episode_paths = set()
        for f in episode_dirs:
            if hasattr(f, "rfilename"):
                if f.rfilename.endswith("full_transcript.json.gz"):
                    ep_path = os.path.dirname(f.rfilename)
                    episode_paths.add(ep_path)
        print(f"Found {len(episode_paths)} episodes/transcripts to download in '{path}'")
        count = 0
        for ep in sorted(episode_paths):
            if count >= episodes_per_dir:
                break
            print(f"Processing episode: {ep}")
            self._process_episode(ep)
            count += 1

    def extract_word_clips(self, words):
        transcripts_dir = self.OUTPUT_TRANSCRIPTS_DIR
        audio_dir = self.OUTPUT_AUDIO_DIR
        output_base = self.OUTPUT_WORD_CLIPS_DIR
        os.makedirs(output_base, exist_ok=True)
        for word in words:
            word_folder = os.path.join(output_base, word)
            os.makedirs(word_folder, exist_ok=True)
        for transcript_file in os.listdir(transcripts_dir):
            if not transcript_file.endswith(".json.gz"):
                continue
            transcript_path = os.path.join(transcripts_dir, transcript_file)
            base_name = transcript_file.replace("_transcript.json.gz", "")
            audio_file = None
            for ext in [".mp3", ".m4a", ".wav", ".flac"]:
                candidate = os.path.join(audio_dir, base_name + ext)
                if os.path.exists(candidate):
                    audio_file = candidate
                    break
            if not audio_file:
                continue
            with gzip.open(transcript_path, "rt", encoding="utf-8") as f:
                try:
                    transcript_json = json.load(f)
                except Exception as e:
                    print(f"Failed to load {transcript_path}: {e}")
                    continue
            try:
                audio = AudioSegment.from_file(audio_file)
            except Exception as e:
                print(f"Failed to load audio {audio_file}: {e}")
                continue
            segments = transcript_json.get("segments", [])
            for seg in segments:
                words_info = seg.get("words", [])
                for word_info in words_info:
                    word_text = word_info.get("word", "").strip().lower()
                    word_start = word_info.get("start", None)
                    word_end = word_info.get("end", None)
                    if word_start is None or word_end is None:
                        continue
                    for target_word in words:
                        if word_text == target_word.lower():
                            word_start_ms = int(float(word_start) * 1000)
                            word_end_ms = int(float(word_end) * 1000)
                            clip_start = max(0, word_start_ms - self.BUFFER_MS)
                            clip_end = min(len(audio), word_end_ms + self.BUFFER_MS)
                            if clip_end > clip_start:
                                clip = audio[clip_start:clip_end]
                                out_folder = os.path.join(output_base, target_word)
                                out_name = f"{base_name}_{word_start_ms}_{word_end_ms}_buffer.wav"
                                out_path = os.path.join(out_folder, out_name)
                                clip.export(out_path, format="wav")
                                print(f"Saved {out_path}")

    def get_random_word_snippets(self, n_per_file=20, exclude_words=None):
        exclude_words = exclude_words or set()
        output_base = os.path.join(self.OUTPUT_WORD_CLIPS_DIR, "random_words")
        os.makedirs(output_base, exist_ok=True)

        transcripts_dir = self.OUTPUT_TRANSCRIPTS_DIR
        audio_dir = self.OUTPUT_AUDIO_DIR

        for transcript_file in os.listdir(transcripts_dir):
            if not transcript_file.endswith(".json.gz"):
                continue

            transcript_path = os.path.join(transcripts_dir, transcript_file)
            base_name = transcript_file.replace("_transcript.json.gz", "")

            # Find corresponding audio file
            audio_file = None
            for ext in [".mp3", ".m4a", ".wav", ".flac"]:
                candidate = os.path.join(audio_dir, base_name + ext)
                if os.path.exists(candidate):
                    audio_file = candidate
                    break

            if not audio_file:
                print(f"No audio file found for {base_name}")
                continue

            # Load transcript
            with gzip.open(transcript_path, "rt", encoding="utf-8") as f:
                try:
                    transcript_json = json.load(f)
                except Exception as e:
                    print(f"Failed to load {transcript_path}: {e}")
                    continue

            # Load audio
            try:
                audio = AudioSegment.from_file(audio_file)
            except Exception as e:
                print(f"Failed to load audio {audio_file}: {e}")
                continue

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
                continue

            # Randomly select up to n_per_file words
            chosen = random.sample(word_occurrences, min(n_per_file, len(word_occurrences)))

            # Save audio snippets
            for idx, (word_text, word_start, word_end) in enumerate(chosen):
                word_start_ms = int(float(word_start) * 1000)
                word_end_ms = int(float(word_end) * 1000)
                clip_start = max(0, word_start_ms - self.BUFFER_MS)
                clip_end = min(len(audio), word_end_ms + self.BUFFER_MS)
                if clip_end > clip_start:
                    clip = audio[clip_start:clip_end]
                    # Sanitize the word_text to make it a valid filename
                    sanitized_word_text = re.sub(r'[^a-zA-Z0-9_\-\u0590-\u05FF]', '_', word_text)
                    out_name = f"{sanitized_word_text}_{word_start_ms}_{word_end_ms}_buffer.wav"
                    out_path = os.path.join(output_base, out_name)
                    clip.export(out_path, format="wav")
                    print(f"Saved {out_path}")

# Example usage
if __name__ == "__main__":
    fetcher = IvritAiDataFetcher()
    for pod in fetcher.list_transcripts_root_folders():
        # Download episodes example
        fetcher.download_episodes(path=pod, episodes_per_dir=5)
        break
    # Extract word clips example
    fetcher.download_episodes(path="PEDSIL", episodes_per_dir=5)
    words_to_extract = ["שלום"]
    fetcher.extract_word_clips(words_to_extract)
    fetcher.get_random_word_snippets(n_per_file=10, exclude_words={"שלום"})
