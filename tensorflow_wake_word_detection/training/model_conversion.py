#!/usr/bin/env python3
"""
Model Conversion for Hebrew Wake Word Detection

This script converts trained TensorFlow models to TensorFlow Lite format
and then to TensorFlow Lite Micro format for deployment on microcontrollers.

Supports quantization for optimal performance on resource-constrained devices.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
from pathlib import Path
import json
from typing import Optional, Dict, Any
import tempfile
import subprocess

class TFLiteConverter:
    """Converter for TensorFlow models to TFLite and TFLite Micro formats."""

    def __init__(self, model_path: str, output_dir: str = "models"):
        """
        Initialize the converter.

        Args:
            model_path: Path to trained Keras model (.h5)
            output_dir: Output directory for converted models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load the model
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Model metadata
        self.input_shape = self.model.input_shape[1:]  # Exclude batch dimension
        self.output_shape = self.model.output_shape[1:]

    def convert_to_tflite(self, quantization: str = "none") -> str:
        """
        Convert Keras model to TensorFlow Lite format.

        Args:
            quantization: Quantization type ('none', 'dynamic', 'int8')

        Returns:
            Path to converted TFLite model
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]

            # For int8 quantization, we need representative dataset
            # This would be provided during actual training
            def representative_dataset_gen():
                # Generate representative dataset for quantization
                # In practice, this should use actual training data
                for _ in range(100):
                    data = np.random.rand(1, *self.input_shape).astype(np.float32)
                    yield [data]

            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # Save TFLite model
        model_name = self.model_path.stem
        tflite_path = self.output_dir / f"{model_name}_{quantization}.tflite"

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {tflite_path}")
        print(f"Model size: {len(tflite_model)} bytes")

        return str(tflite_path)

    def convert_to_tflite_micro(self, tflite_path: str) -> str:
        """
        Convert TFLite model to TFLite Micro C++ header format.

        Args:
            tflite_path: Path to TFLite model

        Returns:
            Path to C++ header file
        """
        tflite_file = Path(tflite_path)

        # Use xxd to convert binary to C array
        header_name = tflite_file.stem.replace('.', '_')
        cpp_path = self.output_dir / f"{header_name}.h"

        try:
            # Convert using xxd (if available) or Python alternative
            with open(tflite_file, 'rb') as f:
                tflite_data = f.read()

            # Convert to C array format
            array_name = f"g_{header_name}"
            array_size = len(tflite_data)

            with open(cpp_path, 'w') as f:
                f.write("#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HEBREW_WAKE_WORD_MODEL_DATA_H_\n")
                f.write("#define TENSORFLOW_LITE_MICRO_EXAMPLES_HEBREW_WAKE_WORD_MODEL_DATA_H_\n")
                f.write("\n")
                f.write("#include <cstdint>\n")
                f.write("\n")
                f.write("extern const unsigned char {}[];\n".format(array_name))
                f.write("extern const unsigned int {}_len;\n".format(array_name))
                f.write("\n")
                f.write("#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_HEBREW_WAKE_WORD_MODEL_DATA_H_\n")

            # Create the corresponding .cc file with the actual data
            cc_path = self.output_dir / f"{header_name}.cc"
            with open(cc_path, 'w') as f:
                f.write('#include "{}"\n'.format(cpp_path.name))
                f.write("\n")
                f.write("// Model data for Hebrew wake word detection\n")
                f.write("const unsigned char {}[] = {{\n".format(array_name))

                # Write data in chunks for readability
                bytes_per_line = 12
                for i in range(0, len(tflite_data), bytes_per_line):
                    chunk = tflite_data[i:i+bytes_per_line]
                    hex_values = [f"0x{b:02x}" for b in chunk]
                    f.write("  " + ", ".join(hex_values))
                    if i + bytes_per_line < len(tflite_data):
                        f.write(",")
                    f.write("\n")

                f.write("};\n")
                f.write("\n")
                f.write("const unsigned int {}_len = {};\n".format(array_name, array_size))

            print(f"TFLite Micro model saved to:")
            print(f"- Header: {cpp_path}")
            print(f"- Source: {cc_path}")

            return str(cpp_path)

        except Exception as e:
            print(f"Error converting to TFLite Micro format: {e}")
            return None

    def generate_model_info(self, tflite_path: str) -> Dict[str, Any]:
        """
        Generate model information and metadata.

        Args:
            tflite_path: Path to TFLite model

        Returns:
            Dictionary with model information
        """
        # Load TFLite model to get metadata
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        model_info = {
            'model_path': str(tflite_path),
            'input_shape': input_details[0]['shape'].tolist(),
            'input_type': str(input_details[0]['dtype']),
            'output_shape': output_details[0]['shape'].tolist(),
            'output_type': str(output_details[0]['dtype']),
            'model_size_bytes': os.path.getsize(tflite_path),
            'tensor_details': {
                'inputs': [{
                    'name': inp['name'],
                    'shape': inp['shape'].tolist(),
                    'dtype': str(inp['dtype'])
                } for inp in input_details],
                'outputs': [{
                    'name': out['name'],
                    'shape': out['shape'].tolist(),
                    'dtype': str(out['dtype'])
                } for out in output_details]
            }
        }

        return model_info

    def save_model_info(self, model_info: Dict[str, Any], filename: str):
        """Save model information to JSON file."""
        info_path = self.output_dir / filename
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"Model info saved to {info_path}")

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description='Convert Hebrew wake word model to TFLite formats')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained Keras model (.h5)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for converted models')
    parser.add_argument('--quantization', type=str, choices=['none', 'dynamic', 'int8'],
                       default='int8', help='Quantization type for TFLite conversion')
    parser.add_argument('--generate_micro', action='store_true', default=True,
                       help='Generate TFLite Micro C++ files')

    args = parser.parse_args()

    # Initialize converter
    converter = TFLiteConverter(args.model_path, args.output_dir)

    print("Converting model to TensorFlow Lite...")

    # Convert to TFLite
    tflite_path = converter.convert_to_tflite(args.quantization)

    # Generate model info
    model_info = converter.generate_model_info(tflite_path)
    converter.save_model_info(model_info, f"model_info_{args.quantization}.json")

    # Convert to TFLite Micro if requested
    if args.generate_micro:
        print("\nConverting to TensorFlow Lite Micro format...")
        micro_path = converter.convert_to_tflite_micro(tflite_path)

        if micro_path:
            # Generate micro model info
            micro_info = {
                'header_file': micro_path,
                'source_file': micro_path.replace('.h', '.cc'),
                'array_name': f"g_{Path(tflite_path).stem.replace('.', '_')}",
                **model_info
            }
            converter.save_model_info(micro_info, f"micro_model_info_{args.quantization}.json")

    print("\nConversion complete!")
    print(f"Original model: {args.model_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"Model size: {model_info['model_size_bytes']} bytes")
    print(f"Input shape: {model_info['input_shape']}")
    print(f"Output shape: {model_info['output_shape']}")

    if args.quantization == 'int8':
        print("\nNote: int8 quantization requires representative dataset.")
        print("For production use, ensure proper calibration data is used.")

if __name__ == "__main__":
    main()
