#!/usr/bin/env python3
"""
Test Setup Script for Custom Wake Word Training

This script verifies that all dependencies are installed and
the training environment is properly configured.

Usage:
    python test_setup.py
"""

import sys
import importlib
from pathlib import Path

class SetupTester:
    """Test the setup and dependencies for custom wake word training."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def test_import(self, module_name, description, critical=True):
        """Test importing a module."""
        try:
            importlib.import_module(module_name)
            print(f"✅ {description}")
            self.passed += 1
            return True
        except ImportError as e:
            if critical:
                print(f"❌ {description} - CRITICAL: {e}")
                self.failed += 1
                return False
            else:
                print(f"⚠️  {description} - WARNING: {e}")
                self.warnings += 1
                return False

    def test_path(self, path, description):
        """Test if a path exists."""
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✅ {description}: {path}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}: {path} - PATH NOT FOUND")
            self.failed += 1
            return False

    def test_script_executable(self, script_path, description):
        """Test if a script exists and is executable."""
        script = Path(script_path)
        if script.exists() and script.is_file():
            print(f"✅ {description}: {script}")
            self.passed += 1
            return True
        else:
            print(f"❌ {description}: {script} - SCRIPT NOT FOUND")
            self.failed += 1
            return False

    def run_comprehensive_test(self):
        """Run all setup tests."""
        print("🧪 Testing Custom Wake Word Training Setup")
        print("=" * 50)

        # Test Python version
        python_version = sys.version_info
        if python_version >= (3, 7):
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            self.passed += 1
        else:
            print(f"❌ Python version: {python_version.major}.{python_version.minor}.{python_version.micro} - Need Python 3.7+")
            self.failed += 1

        print("\n📦 Testing Dependencies...")

        # Critical dependencies
        self.test_import("pydub", "PyDub (audio processing)")
        self.test_import("librosa", "Librosa (audio analysis)")
        self.test_import("tensorflow", "TensorFlow (machine learning)")
        self.test_import("sklearn", "Scikit-learn (data processing)")
        self.test_import("numpy", "NumPy (numerical computing)")
        self.test_import("pandas", "Pandas (data handling)")
        self.test_import("tqdm", "TQDM (progress bars)")
        self.test_import("yaml", "PyYAML (configuration)")

        # Optional dependencies
        self.test_import("pyaudio", "PyAudio (audio recording)", critical=False)
        self.test_import("matplotlib", "Matplotlib (plotting)", critical=False)
        self.test_import("seaborn", "Seaborn (advanced plotting)", critical=False)

        print("\n📁 Testing File Structure...")

        # Test current directory structure
        self.test_path("CustomSoundSamples", "Custom sound samples directory")
        self.test_path("models", "Models output directory")
        self.test_path("processed_data", "Processed data directory")

        # Check for training scripts
        self.test_path("train_custom_word.py", "Main training script")
        self.test_path("record_samples.py", "Recording script")
        self.test_path("requirements.txt", "Requirements file")

        print("\n🔗 Testing External Scripts...")

        # Test paths to external processing scripts
        project_root = Path(__file__).parent.parent

        self.test_script_executable(
            project_root / "audio_prosser" / "audio_processor.py",
            "Audio processor script"
        )
        self.test_script_executable(
            project_root / "audio_prosser" / "noise_adder.py",
            "Noise adder script"
        )
        self.test_script_executable(
            project_root / "tensorflow_wake_word_detection" / "training" / "data_preprocessing.py",
            "Data preprocessing script"
        )
        self.test_script_executable(
            project_root / "tensorflow_wake_word_detection" / "training" / "model_training.py",
            "Model training script"
        )

        print("\n🧠 Testing TensorFlow...")

        # Test TensorFlow functionality
        try:
            import tensorflow as tf
            print(f"   TensorFlow version: {tf.__version__}")

            # Test basic TF operation
            hello = tf.constant("Hello, TensorFlow!")
            print(f"   Basic TF test: {hello.numpy().decode('utf-8')}")

            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✅ GPU available: {len(gpus)} device(s)")
            else:
                print("   ℹ️  No GPU detected (CPU training will be slower but still works)")

            self.passed += 1

        except Exception as e:
            print(f"❌ TensorFlow functionality test failed: {e}")
            self.failed += 1

        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.passed}")
        if self.warnings > 0:
            print(f"⚠️  Warnings: {self.warnings}")
        if self.failed > 0:
            print(f"❌ Failed: {self.failed}")

        if self.failed == 0:
            print("\n🎉 All critical tests passed! You're ready to train custom wake words.")
            print("\n🚀 Quick start:")
            print("   1. python record_samples.py 'Hello' 5")
            print("   2. python train_custom_word.py Hello")
        else:
            print("❌ Some critical components are missing. Please fix the failed tests.")            
            print("   Run: pip install -r requirements.txt")

        return self.failed == 0

def main():
    """Main test function."""
    tester = SetupTester()
    success = tester.run_comprehensive_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
