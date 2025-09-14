#!/usr/bin/env python3
"""
Simple test script to validate the audio processor structure and directory discovery.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from audio_processor import AudioProcessor, AudioProcessorError
    print("✓ Successfully imported AudioProcessor")

    # Test directory discovery
    test_dir = Path("SoundSamples")
    if test_dir.exists():
        print(f"✓ Found SoundSamples directory: {test_dir.absolute()}")

        # Create a mock processor to test directory discovery
        try:
            processor = AudioProcessor(str(test_dir))
            print("✓ AudioProcessor initialized successfully")

            # Test subfolder discovery
            subfolders = processor.discover_subfolders()
            print(f"✓ Discovered {len(subfolders)} subfolders:")
            for subfolder in subfolders:
                print(f"  - {subfolder.name}")

                # Test audio file discovery
                audio_files = processor.discover_audio_files(subfolder)
                print(f"    Contains {len(audio_files)} WAV files")

        except AudioProcessorError as e:
            print(f"✗ AudioProcessor error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    else:
        print(f"✗ SoundSamples directory not found at: {test_dir.absolute()}")

except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\nTest completed!")
