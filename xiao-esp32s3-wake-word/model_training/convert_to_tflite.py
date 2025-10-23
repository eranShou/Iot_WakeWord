"""
Convert trained Keras model to TFLite format
Float32 conversion without quantization for ESP32-S3 compatibility
All parameters loaded from config.json - no magic numbers
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def convert_to_tflite():
    """
    Convert trained Keras model to TFLite format
    """
    config = load_config()
    
    print("Converting model to TFLite...")
    print("=" * 50)
    
    # Load trained model
    keras_model_path = config['output']['keras_model']
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")
    
    print(f"Loading Keras model from: {keras_model_path}")
    model = keras.models.load_model(keras_model_path)
    
    # Print model info
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure converter (float32, no quantization)
    converter.optimizations = []  # No optimizations
    converter.target_spec.supported_types = [tf.float32]  # Float32 only
    
    print("Converting to TFLite (float32, no quantization)...")
    
    # Convert model
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    
    # Save TFLite model
    tflite_path = config['output']['tflite_model']
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print model info
    model_size_kb = len(tflite_model) / 1024
    print(f"\nTFLite model saved to: {tflite_path}")
    print(f"Model size: {model_size_kb:.2f} KB")
    
    if model_size_kb > 500:
        print("⚠ Warning: Model size exceeds 500KB target")
        print("Consider reducing model architecture in config.json")
    elif model_size_kb > 100:
        print("⚠ Warning: Model size exceeds 100KB for optimal ESP32-S3 performance")
        print("Model will work but may be slow on ESP32-S3")
    else:
        print("✓ Model size is within ESP32-S3 limits")
    
    # Test inference with sample data
    print("\nTesting TFLite inference...")
    test_inference(tflite_model, config)
    
    return tflite_path, model_size_kb

def test_inference(tflite_model, config):
    """
    Test TFLite model inference with sample data
    """
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Create sample input (random spectrogram)
    input_shape = input_details[0]['shape']
    sample_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], sample_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(output).numpy()
    
    print(f"Sample inference output:")
    classes = config['classes']['labels']
    for i, class_name in enumerate(classes):
        print(f"  {class_name}: {probabilities[0][i]:.4f}")
    
    # Find predicted class
    predicted_class = np.argmax(probabilities[0])
    confidence = probabilities[0][predicted_class]
    
    print(f"Predicted class: {classes[predicted_class]} (confidence: {confidence:.4f})")

def verify_model_compatibility():
    """
    Verify TFLite model is compatible with ESP32-S3
    """
    config = load_config()
    
    print("\nVerifying ESP32-S3 compatibility...")
    
    # Check model size
    tflite_path = config['output']['tflite_model']
    if os.path.exists(tflite_path):
        model_size = os.path.getsize(tflite_path)
        model_size_kb = model_size / 1024
        
        print(f"Model size: {model_size_kb:.2f} KB")
        
        if model_size_kb <= 100:
            print("✓ Model size is suitable for ESP32-S3")
        else:
            print("⚠ Model size may be too large for ESP32-S3")
        
        # Check tensor arena size
        tensor_arena_size = config['esp32']['tensor_arena_size']
        print(f"Tensor arena size: {tensor_arena_size} bytes")
        
        if tensor_arena_size >= 8000:
            print("✓ Tensor arena size is sufficient")
        else:
            print("⚠ Tensor arena size may be too small")
    
    else:
        print("Error: TFLite model not found")

if __name__ == "__main__":
    try:
        tflite_path, model_size = convert_to_tflite()
        verify_model_compatibility()
        print(f"\nTFLite conversion completed successfully!")
        print(f"Model saved to: {tflite_path}")
        print(f"Model size: {model_size:.2f} KB")
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
