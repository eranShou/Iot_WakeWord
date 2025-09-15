#!/usr/bin/env python3
"""
Quick Start Script for Custom Wake Word Training

This script provides the fastest way to get started with custom wake word training.
It combines recording and training into a single streamlined process.

Usage:
    python quick_start.py <word_name>

Example:
    python quick_start.py Hello

This will:
1. Record 5 audio samples
2. Train the model
3. Provide usage instructions
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def quick_start(word_name):
    """Complete quick start workflow."""
    print(f"🚀 Quick Start: Training custom wake word '{word_name}'")
    print("=" * 60)

    # Step 1: Test setup
    print("\n🧪 Step 1: Testing Setup")
    if not run_command("python test_setup.py", "Testing environment setup"):
        print("❌ Setup test failed. Please fix issues and try again.")
        return False

    # Step 2: Record samples
    print("\n🎤 Step 2: Recording Audio Samples")
    record_cmd = f'python record_samples.py "{word_name}" 5'
    if not run_command(record_cmd, "Recording 5 audio samples"):
        print("❌ Audio recording failed.")
        return False

    # Step 3: Train model
    print("\n🤖 Step 3: Training Model")
    train_cmd = f'python train_custom_word.py "{word_name}"'
    if not run_command(train_cmd, "Training wake word model"):
        print("❌ Model training failed.")
        return False

    # Success!
    print("\n" + "=" * 60)
    print("🎉 QUICK START COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully trained wake word: '{word_name}'")
    print("📁 Your model files are in the 'models/' folder:")
    print("   • hebrew_wake_word_model_cnn.h5 (use this in your microcontroller)")
    print("   • training_results_cnn.json (performance metrics)")
    print("   • custom_word_{word_name}_summary.json (usage instructions)")
    print("🚀 Next steps:")
    print("   1. Copy the .h5 model file to your microcontroller project")
    print("   2. Update your model loading code")    
    print("   3. Test the wake word detection!")
    return True

def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(
        description="Quick start custom wake word training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py Hello
  python quick_start.py "Hey Computer"

This script will:
1. Test your setup
2. Record 5 audio samples
3. Train the model automatically
4. Provide next steps

Make sure your microphone is ready before starting!
        """
    )

    parser.add_argument(
        "word_name",
        help="Name of the wake word to train"
    )

    args = parser.parse_args()

    # Validate word name
    word_name = args.word_name.strip()
    if not word_name:
        print("❌ Error: Word name cannot be empty")
        return 1

    # Run quick start
    success = quick_start(word_name)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
