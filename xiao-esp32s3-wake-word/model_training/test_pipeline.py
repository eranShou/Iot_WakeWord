"""
Test script to verify the enhanced pipeline works correctly
Tests dataset loading, model architecture, and size constraints
"""

import json
import os
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def test_data_paths():
    """
    Test that all required data paths exist
    """
    print("Testing data paths...")
    print("=" * 30)
    
    config = load_config()
    data_paths = config['data_paths']
    
    all_exist = True
    
    for path_name, path_value in data_paths.items():
        if os.path.exists(path_value):
            # Count files in directory
            wav_files = list(Path(path_value).glob('*.wav'))
            print(f"✓ {path_name}: {len(wav_files)} files")
        else:
            print(f"✗ {path_name}: Path not found")
            all_exist = False
    
    return all_exist

def test_model_architecture():
    """
    Test that model architecture is optimized for size
    """
    print("\nTesting model architecture...")
    print("=" * 30)
    
    config = load_config()
    model_config = config['model']
    
    # Check if architecture is reduced
    conv1_filters = model_config['conv1_filters']
    conv2_filters = model_config['conv2_filters']
    dense_units = model_config['dense_units']
    
    print(f"Conv1 filters: {conv1_filters}")
    print(f"Conv2 filters: {conv2_filters}")
    print(f"Dense units: {dense_units}")
    
    # Check if architecture is small enough
    if conv1_filters <= 16 and conv2_filters <= 32 and dense_units <= 64:
        print("✓ Model architecture is optimized for small size")
        return True
    else:
        print("⚠ Model architecture may be too large")
        return False

def test_configuration():
    """
    Test that configuration is valid
    """
    print("\nTesting configuration...")
    print("=" * 30)
    
    try:
        config = load_config()
        
        # Check required sections
        required_sections = ['audio', 'spectrogram', 'model', 'training', 'classes', 'data_paths', 'output', 'esp32']
        for section in required_sections:
            if section in config:
                print(f"✓ {section}")
            else:
                print(f"✗ {section} missing")
                return False
        
        # Check model size target
        esp32_config = config['esp32']
        if 'tensor_arena_size' in esp32_config:
            print(f"✓ Tensor arena size: {esp32_config['tensor_arena_size']}")
        
        print("✓ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_ivrit_ai_integration():
    """
    Test that ivrit-ai data paths are configured
    """
    print("\nTesting ivrit-ai integration...")
    print("=" * 30)
    
    config = load_config()
    data_paths = config['data_paths']
    
    ivrit_paths = [
        'train_ivrit_lehitraoot',
        'train_ivrit_shaloom'
    ]
    
    for path_name in ivrit_paths:
        if path_name in data_paths:
            path_value = data_paths[path_name]
            print(f"✓ {path_name}: {path_value}")
        else:
            print(f"✗ {path_name}: Not configured")
            return False
    
    print("✓ ivrit-ai integration configured")
    return True

def main():
    """
    Run all tests
    """
    print("Enhanced Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Model Architecture", test_model_architecture),
        ("Data Paths", test_data_paths),
        ("ivrit-ai Integration", test_ivrit_ai_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("Test Summary:")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Enhanced pipeline is ready for training!")
    else:
        print("\n✗ Pipeline needs fixes before training.")
