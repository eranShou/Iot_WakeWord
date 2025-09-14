#!/usr/bin/env python3
"""
System Testing and Validation for Hebrew Wake Word Detection

This script provides comprehensive testing and validation of the complete
Hebrew wake word detection system, including model evaluation, audio
processing validation, and performance benchmarking.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
from pathlib import Path
import librosa
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Import our custom modules
import sys
sys.path.append('..')

try:
    from training.data_preprocessing import HebrewWakeWordPreprocessor
    from training.model_training import HebrewWakeWordModel
except ImportError:
    print("Warning: Could not import training modules. Some tests may be skipped.")

class SystemTester:
    """Comprehensive system testing and validation."""

    def __init__(self, model_path: str, data_path: str, output_dir: str = "test_results"):
        """
        Initialize system tester.

        Args:
            model_path: Path to trained model
            data_path: Path to processed data
            output_dir: Output directory for test results
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None

        # Load test data
        try:
            data = np.load(data_path)
            self.X_test = data['X_test']
            self.y_test = data['y_test']
            self.label_names = data['label_names']
            print(f"✓ Test data loaded: {len(self.X_test)} samples")
        except Exception as e:
            print(f"✗ Failed to load test data: {e}")
            self.X_test = None

    def run_model_evaluation(self) -> Dict:
        """Run comprehensive model evaluation."""
        if self.model is None or self.X_test is None:
            return {"error": "Model or test data not available"}

        print("\n=== Model Evaluation ===")

        # Get predictions
        start_time = time.time()
        y_pred_prob = self.model.predict(self.X_test, verbose=0)
        inference_time = time.time() - start_time

        y_pred = np.argmax(y_pred_prob, axis=1)

        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Per-class accuracy
        per_class_accuracy = {}
        for i, label in enumerate(self.label_names):
            class_mask = (self.y_test == i)
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == i)
                per_class_accuracy[label] = class_accuracy

        results = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "inference_time": float(inference_time),
            "avg_inference_time_per_sample": float(inference_time / len(self.X_test)),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": per_class_accuracy,
            "total_samples": len(self.X_test)
        }

        # Print results
        print(".4f")
        print(".4f")
        print(".2f")
        print("Per-class accuracy:")
        for label, acc in per_class_accuracy.items():
            print(".4f")

        return results

    def test_audio_processing(self) -> Dict:
        """Test audio processing pipeline."""
        print("\n=== Audio Processing Test ===")

        results = {}

        # Test MFCC extraction with sample audio
        try:
            # Create a simple test tone
            sr = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration), False)

            # Generate test signals
            test_signals = {
                "sine_440hz": np.sin(2 * np.pi * 440 * t),
                "noise": np.random.randn(len(t)) * 0.1,
                "silence": np.zeros(len(t))
            }

            for name, audio in test_signals.items():
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(
                    y=audio, sr=sr, n_mfcc=40,
                    n_fft=int(0.03 * sr), hop_length=int(0.01 * sr)
                )

                results[name] = {
                    "mfcc_shape": mfccs.shape,
                    "mfcc_mean": float(np.mean(mfccs)),
                    "mfcc_std": float(np.std(mfccs)),
                    "audio_rms": float(np.sqrt(np.mean(audio**2)))
                }

                print(f"✓ {name}: MFCC shape {mfccs.shape}, RMS {results[name]['audio_rms']:.4f}")

        except Exception as e:
            print(f"✗ Audio processing test failed: {e}")
            results["error"] = str(e)

        return results

    def benchmark_performance(self) -> Dict:
        """Benchmark system performance."""
        print("\n=== Performance Benchmark ===")

        if self.model is None:
            return {"error": "Model not available for benchmarking"}

        results = {}

        # Model size
        model_size = os.path.getsize(self.model_path)
        results["model_size_bytes"] = model_size
        results["model_size_kb"] = model_size / 1024
        results["model_size_mb"] = model_size / (1024 * 1024)

        # Model architecture info
        total_params = self.model.count_params()
        results["total_parameters"] = total_params

        # Layer information
        layer_info = []
        for i, layer in enumerate(self.model.layers):
            layer_info.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output_shape),
                "params": layer.count_params()
            })

        results["layers"] = layer_info

        # Memory estimation for ESP32-S3
        # Rough estimation: 4 bytes per parameter + overhead
        estimated_memory = total_params * 4 + 50 * 1024  # +50KB overhead
        results["estimated_memory_bytes"] = estimated_memory
        results["estimated_memory_kb"] = estimated_memory / 1024

        # Print results
        print(f"Model size: {results['model_size_kb']:.1f} KB")
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated ESP32 memory usage: {results['estimated_memory_kb']:.1f} KB")

        return results

    def test_edge_cases(self) -> Dict:
        """Test edge cases and error handling."""
        print("\n=== Edge Cases Testing ===")

        results = {}

        if self.model is None or self.X_test is None:
            return {"error": "Model or test data not available"}

        # Test with different input shapes
        original_shape = self.X_test.shape[1:]

        # Test with zero input
        zero_input = np.zeros((1,) + original_shape)
        try:
            zero_pred = self.model.predict(zero_input, verbose=0)
            results["zero_input_handling"] = {
                "prediction_shape": zero_pred.shape,
                "max_confidence": float(np.max(zero_pred)),
                "predicted_class": int(np.argmax(zero_pred))
            }
            print(f"✓ Zero input test: Predicted class {results['zero_input_handling']['predicted_class']}")
        except Exception as e:
            results["zero_input_error"] = str(e)
            print(f"✗ Zero input test failed: {e}")

        # Test with random noise
        noise_input = np.random.randn(1, *original_shape)
        try:
            noise_pred = self.model.predict(noise_input, verbose=0)
            results["noise_input_handling"] = {
                "prediction_shape": noise_pred.shape,
                "max_confidence": float(np.max(noise_pred)),
                "predicted_class": int(np.argmax(noise_pred))
            }
            print(f"✓ Noise input test: Predicted class {results['noise_input_handling']['predicted_class']}")
        except Exception as e:
            results["noise_input_error"] = str(e)
            print(f"✗ Noise input test failed: {e}")

        # Test confidence distribution
        y_pred_prob = self.model.predict(self.X_test, verbose=0)
        confidences = np.max(y_pred_prob, axis=1)

        results["confidence_stats"] = {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "median_confidence": float(np.median(confidences))
        }

        print("Confidence statistics:")
        print(".4f")
        print(".4f")

        return results

    def _safe_format_number(self, value, default="N/A", format_spec=".4f"):
        """Safely format a number, handling string defaults."""
        if isinstance(value, (int, float)):
            return f"{value:{format_spec}}"
        return str(value) if value != default else default

    def generate_reports(self, all_results: Dict):
        """Generate comprehensive test reports."""
        print("\n=== Generating Test Reports ===")

        # Save detailed results
        import json
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✓ Detailed results saved to {results_file}")

        # Generate summary report
        summary_file = self.output_dir / "test_summary.md"

        with open(summary_file, 'w') as f:
            f.write("# Hebrew Wake Word Detection - Test Summary\n\n")

            if "model_evaluation" in all_results:
                eval_results = all_results["model_evaluation"]
                f.write("## Model Evaluation\n\n")
                test_accuracy = self._safe_format_number(eval_results.get('test_accuracy'), format_spec=".4f")
                f.write(f"- Test Accuracy: {test_accuracy}\n")
                test_loss = self._safe_format_number(eval_results.get('test_loss'), format_spec=".4f")
                f.write(f"- Test Loss: {test_loss}\n")
                inference_time = self._safe_format_number(eval_results.get('inference_time'), format_spec=".2f")
                f.write(f"- Inference time: {inference_time}s\n")
                f.write(f"- Samples tested: {eval_results.get('total_samples', 'N/A')}\n\n")

                if "per_class_accuracy" in eval_results:
                    f.write("### Per-Class Accuracy\n\n")
                    for label, acc in eval_results["per_class_accuracy"].items():
                        acc_formatted = self._safe_format_number(acc, format_spec=".4f")
                        f.write(f"- {label}: {acc_formatted}\n")
                    f.write("\n")

            if "performance" in all_results:
                perf_results = all_results["performance"]
                f.write("## Performance Metrics\n\n")
                model_size_kb = self._safe_format_number(perf_results.get('model_size_kb'), format_spec=".1f")
                f.write(f"- Model size: {model_size_kb} KB\n")
                total_params = perf_results.get('total_parameters', 'N/A')
                if isinstance(total_params, (int, float)):
                    f.write(f"- Total parameters: {total_params:,}\n")
                else:
                    f.write(f"- Total parameters: {total_params}\n")
                estimated_memory_kb = self._safe_format_number(perf_results.get('estimated_memory_kb'), format_spec=".1f")
                f.write(f"- Estimated ESP32 memory usage: {estimated_memory_kb} KB\n")
                f.write("\n")

            if "edge_cases" in all_results:
                edge_results = all_results["edge_cases"]
                f.write("## Edge Cases\n\n")
                if "confidence_stats" in edge_results:
                    conf_stats = edge_results["confidence_stats"]
                    mean_conf = self._safe_format_number(conf_stats.get('mean_confidence'), format_spec=".4f")
                    f.write(f"- Mean confidence: {mean_conf}\n")
                    std_conf = self._safe_format_number(conf_stats.get('std_confidence'), format_spec=".4f")
                    f.write(f"- Confidence std dev: {std_conf}\n")
                    f.write("\n")

            f.write("## Recommendations\n\n")

            # Generate recommendations based on results
            if "model_evaluation" in all_results:
                accuracy = all_results["model_evaluation"].get("test_accuracy", 0)
                if accuracy > 0.9:
                    f.write("- [SUCCESS] Excellent model performance!\n")
                elif accuracy > 0.8:
                    f.write("- [GOOD] Good model performance\n")
                else:
                    f.write("- [WARNING] Consider retraining model with more data or different architecture\n")

            if "performance" in all_results:
                mem_kb = all_results["performance"].get("estimated_memory_kb", 1000)
                if mem_kb > 200:
                    f.write("- [WARNING] High memory usage - consider model optimization\n")
                else:
                    f.write("- [SUCCESS] Memory usage within ESP32-S3 limits\n")

            f.write("\n## Next Steps\n\n")
            f.write("1. Deploy model to ESP32-S3 microcontroller\n")
            f.write("2. Test real-time wake word detection\n")
            f.write("3. Optimize for power consumption if needed\n")
            f.write("4. Consider adding custom wake words\n")

        print(f"✓ Summary report saved to {summary_file}")

    def run_all_tests(self) -> Dict:
        """Run all system tests."""
        print("=== Starting Comprehensive System Testing ===")

        all_results = {}

        # Run individual tests
        all_results["model_evaluation"] = self.run_model_evaluation()
        all_results["audio_processing"] = self.test_audio_processing()
        all_results["performance"] = self.benchmark_performance()
        all_results["edge_cases"] = self.test_edge_cases()

        # Generate reports
        self.generate_reports(all_results)

        print("\n=== System Testing Complete ===")
        return all_results

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test Hebrew wake word detection system')
    parser.add_argument('--model_path', type=str, default='models/hebrew_wake_word_model_cnn.h5',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='processed_data/hebrew_wake_word_data.npz',
                       help='Path to processed data')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Output directory for test results')
    parser.add_argument('--test_type', type=str, choices=['all', 'model', 'audio', 'performance', 'edge'],
                       default='all', help='Type of test to run')

    args = parser.parse_args()

    # Initialize tester
    tester = SystemTester(args.model_path, args.data_path, args.output_dir)

    # Run tests based on type
    if args.test_type == 'all':
        results = tester.run_all_tests()
    elif args.test_type == 'model':
        results = {"model_evaluation": tester.run_model_evaluation()}
    elif args.test_type == 'audio':
        results = {"audio_processing": tester.test_audio_processing()}
    elif args.test_type == 'performance':
        results = {"performance": tester.benchmark_performance()}
    elif args.test_type == 'edge':
        results = {"edge_cases": tester.test_edge_cases()}

    # Print final summary
    print("\n=== Test Results Summary ===")

    if "model_evaluation" in results:
        eval_res = results["model_evaluation"]
        if "test_accuracy" in eval_res:
            test_accuracy = tester._safe_format_number(eval_res.get("test_accuracy"), format_spec=".4f")
            print(f"Model Accuracy: {test_accuracy}")
        if "avg_inference_time_per_sample" in eval_res:
            avg_time = tester._safe_format_number(eval_res.get("avg_inference_time_per_sample"), format_spec=".2f")
            print(f"Average Inference Time: {avg_time} ms")

    if "performance" in results:
        perf_res = results["performance"]
        if "model_size_kb" in perf_res:
            model_size = tester._safe_format_number(perf_res.get("model_size_kb"), format_spec=".1f")
            print(f"Model Size: {model_size} KB")
        if "estimated_memory_kb" in perf_res:
            mem_usage = tester._safe_format_number(perf_res.get("estimated_memory_kb"), format_spec=".1f")
            print(f"Estimated Memory Usage: {mem_usage} KB")

    print("Test results saved in:", args.output_dir)
    print("✓ System testing completed!")

if __name__ == "__main__":
    main()
