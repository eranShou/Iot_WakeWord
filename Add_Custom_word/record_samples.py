#!/usr/bin/env python3
"""
Simple Audio Recording Script for Custom Wake Words

This script helps you record audio samples for custom wake word training.
It records short audio clips and saves them as WAV files.

Usage:
    python record_samples.py <word_name> <num_samples>

Example:
    python record_samples.py Hello 5

Requirements:
    pip install pyaudio wave

On Windows, you might need to install portaudio:
    pip install pipwin
    pipwin install pyaudio
"""

import pyaudio
import wave
import sys
import time
from pathlib import Path
import argparse

class AudioRecorder:
    """Simple audio recorder for wake word samples."""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate (16kHz for wake word detection)
            channels: Number of audio channels (1 = mono)
            chunk_size: Audio buffer chunk size
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        self.pyaudio = None
        self.stream = None

    def initialize_audio(self):
        """Initialize PyAudio."""
        try:
            self.pyaudio = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"❌ Failed to initialize audio: {e}")
            print("   Make sure pyaudio is installed:")
            print("   pip install pyaudio")
            return False

    def open_stream(self):
        """Open audio stream."""
        try:
            self.stream = self.pyaudio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            return True
        except Exception as e:
            print(f"❌ Failed to open audio stream: {e}")
            print("   Check that your microphone is connected and not in use by another application.")
            return False

    def record_sample(self, duration_seconds=1.0):
        """
        Record an audio sample.

        Args:
            duration_seconds: Recording duration in seconds

        Returns:
            Audio frames as bytes
        """
        print(f"   🎤 Recording for {duration_seconds} second{'s' if duration_seconds != 1 else ''}...")

        frames = []
        total_chunks = int(self.sample_rate / self.chunk_size * duration_seconds)

        for i in range(total_chunks):
            if self.stream.is_active():
                data = self.stream.read(self.chunk_size)
                frames.append(data)
            else:
                break

        return b''.join(frames)

    def save_wav(self, frames, filename):
        """
        Save audio frames to WAV file.

        Args:
            frames: Audio frames
            filename: Output filename
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(frames)

    def close(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.pyaudio:
            self.pyaudio.terminate()

def countdown(seconds):
    """Display countdown timer."""
    for i in range(seconds, 0, -1):
        print(f"   Starting recording in {i}...", end='\r')
        time.sleep(1)
    print("   🎯 Go! Say your wake word now.")

def main():
    """Main recording function."""
    parser = argparse.ArgumentParser(
        description="Record audio samples for custom wake word training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record_samples.py Hello 5
  python record_samples.py "Hey Computer" 3

Instructions:
1. Make sure your microphone is connected and working
2. Run this script with your word name and number of samples
3. Follow the prompts to record each sample
4. Samples will be saved in CustomSoundSamples/YourWordName/

Note: Each recording should be about 1 second of saying your wake word.
        """
    )

    parser.add_argument(
        "word_name",
        help="Name of the wake word to record"
    )

    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of audio samples to record"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Recording duration in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before recording (default: 3)"
    )

    args = parser.parse_args()

    # Validate inputs
    word_name = args.word_name.strip()
    if not word_name:
        print("❌ Error: Word name cannot be empty")
        return 1

    if args.num_samples < 1:
        print("❌ Error: Number of samples must be at least 1")
        return 1

    # Create output directory
    output_dir = Path("CustomSoundSamples") / word_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎤 Recording {args.num_samples} samples for wake word: '{word_name}'")
    print(f"📁 Samples will be saved to: {output_dir}")
    print("=" * 60)

    # Initialize recorder
    recorder = AudioRecorder()

    if not recorder.initialize_audio():
        return 1

    if not recorder.open_stream():
        recorder.close()
        return 1

    try:
        # Record samples
        for i in range(args.num_samples):
            sample_num = i + 1
            filename = output_dir / "02d"

            print(f"\n🎵 Sample {sample_num}/{args.num_samples}")
            print(f"   File: {filename}")

            # Countdown
            countdown(args.countdown)

            # Record
            frames = recorder.record_sample(args.duration)

            # Save
            recorder.save_wav(frames, str(filename))
            print(f"   ✅ Saved: {filename}")

            # Brief pause between recordings
            if sample_num < args.num_samples:
                print("   📋 Get ready for the next recording...")
                time.sleep(2)

        print("\n" + "=" * 60)
        print("🎉 Recording complete!")
        print(f"✅ Recorded {args.num_samples} samples for '{word_name}'")
        print(f"📁 Files saved in: {output_dir}")
        print("\n🚀 Next step: Run training")
        print(f"   python train_custom_word.py \"{word_name}\"")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⏹️  Recording interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Recording failed: {e}")
        return 1

    finally:
        recorder.close()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Recording interrupted")
        sys.exit(1)
