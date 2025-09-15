#!/usr/bin/env python3
"""
Example Usage of Custom Wake Word Training

This script demonstrates how to use the custom wake word training system.
It creates sample data and shows the complete workflow.

Usage:
    python example_usage.py
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import wave
import struct

class ExampleGenerator:
    """Generate example audio files for testing the training system."""

    def __init__(self):
        self.sample_rate = 16000
        self.duration = 1.0  # seconds
        self.frequency = 440  # A note in Hz

    def generate_tone(self, frequency, duration, sample_rate):
        """Generate a sine wave tone."""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Fade in/out to avoid clicks
        fade_len = int(sample_rate * 0.1)  # 100ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        # Generate tone
        tone = np.sin(frequency * 2 * np.pi * t)

        # Apply fades
        tone[:fade_len] *= fade_in
        tone[-fade_len:] *= fade_out

        return tone

    def create_wav_file(self, filename, audio_data, sample_rate):
        """Create a WAV file from audio data."""
        # Normalize to 16-bit range
        audio_data = np.int16(audio_data * 32767)

        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

    def generate_example_word(self, word_name, variations=5):
        """
        Generate example audio files for a wake word.

        Args:
            word_name: Name of the word
            variations: Number of different variations to generate
        """
        output_dir = Path("CustomSoundSamples") / word_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎵 Generating {variations} example recordings for '{word_name}'")

        for i in range(variations):
            # Create slight variations in frequency and duration
            freq_variation = self.frequency + (i - 2) * 20  # ±40 Hz variation
            duration_variation = self.duration + (i - 2) * 0.05  # ±0.1s variation

            # Generate tone
            audio = self.generate_tone(freq_variation, duration_variation, self.sample_rate)

            # Create filename
            filename = output_dir / "02d"
            self.create_wav_file(str(filename), audio, self.sample_rate)

            print(f"   Created: {filename}")

        print(f"✅ Example files saved to: {output_dir}")

def demonstrate_workflow():
    """Demonstrate the complete custom wake word training workflow."""
    print("🚀 Custom Wake Word Training - Example Workflow")
    print("=" * 60)

    # Step 1: Generate example data
    print("\n📝 Step 1: Generating Example Audio Data")
    generator = ExampleGenerator()
    generator.generate_example_word("Hello", 5)

    # Step 2: Show the file structure
    print("\n📁 Step 2: Checking File Structure")
    example_dir = Path("CustomSoundSamples/Hello")
    if example_dir.exists():
        wav_files = list(example_dir.glob("*.wav"))
        print(f"   Found {len(wav_files)} audio files:")
        for wav_file in wav_files:
            print(f"   • {wav_file.name}")
    else:
        print("   ❌ Example directory not found!")
        return False

    # Step 3: Show training command
    print("\n🎯 Step 3: Training Command")
    print("   To train the model, run:")
    print("   python train_custom_word.py Hello")
    print("   ")
    print("   Or with options:")
    print("   python train_custom_word.py Hello --skip-noise")
    print("   python train_custom_word.py Hello --keep-intermediates")

    # Step 4: Expected outputs
    print("\n📤 Step 4: Expected Outputs")
    print("   After training completes, you'll find:")
    print("   • models/hebrew_wake_word_model_cnn.h5 (trained model)")
    print("   • models/training_results_cnn.json (performance metrics)")
    print("   • models/custom_word_Hello_summary.json (usage info)")
    print("   • Training plots and confusion matrix (if enabled)")

    print("\n" + "=" * 60)
    print("🎉 Example setup complete!")
    print("=" * 60)
    print("Now try running the training:")
    print("   python train_custom_word.py Hello")
    print("\nOr test the setup first:")
    print("   python test_setup.py")

    return True

def cleanup_example():
    """Clean up example files."""
    example_dir = Path("CustomSoundSamples/Hello")
    if example_dir.exists():
        print(f"🧹 Cleaning up example files in {example_dir}")
        shutil.rmtree(example_dir)

def main():
    """Main example function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_example()
        return 0

    try:
        success = demonstrate_workflow()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
