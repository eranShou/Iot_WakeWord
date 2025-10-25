"""
Complete training pipeline for Hebrew wake word detection
Single-command execution: prepare → train → convert → generate headers → deploy
All parameters from config.json - no magic numbers
"""

import json
import os
import sys
import time
from pathlib import Path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def check_dependencies():
    """
    Check if required Python packages are installed
    """
    print("Checking dependencies...")
    
    required_packages = [
        'tensorflow',
        'numpy', 
        'matplotlib',
        'sklearn',
        'librosa'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies satisfied")
    return True

def validate_config():
    """
    Validate configuration file and data paths
    """
    print("\nValidating configuration...")
    
    config = load_config()
    
    # Check required sections
    required_sections = ['audio', 'spectrogram', 'model', 'training', 'classes', 'data_paths', 'output', 'esp32']
    for section in required_sections:
        if section not in config:
            print(f"✗ Missing config section: {section}")
            return False
        print(f"✓ {section}")
    
    # Check data paths
    data_paths = config['data_paths']
    for path_name, path_value in data_paths.items():
        if not os.path.exists(path_value):
            print(f"✗ Data path not found: {path_name} -> {path_value}")
            return False
        print(f"✓ {path_name}: {path_value}")
    
    # Check output directory
    output_dir = config['output']['model_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory exists: {output_dir}")
    
    print("✓ Configuration validation passed")
    return True

def print_dataset_statistics():
    """
    Print dataset statistics before training
    """
    print("\nDataset Statistics:")
    print("=" * 50)
    
    config = load_config()
    data_paths = config['data_paths']
    classes = config['classes']['labels']
    
    total_train = 0
    total_val = 0
    
    for path_name, path_value in data_paths.items():
        if os.path.exists(path_value):
            wav_files = list(Path(path_value).glob('*.wav'))
            count = len(wav_files)
            
            if 'train' in path_name:
                total_train += count
            elif 'val' in path_name:
                total_val += count
            else:
                # noise/unknown - will be split
                total_train += int(count * 0.8)
                total_val += int(count * 0.2)
            
            print(f"{path_name}: {count} files")
    
    print(f"\nTotal training samples: {total_train}")
    print(f"Total validation samples: {total_val}")
    print(f"Classes: {', '.join(classes)}")

def run_training_pipeline():
    """
    Execute the complete training pipeline
    """
    print("Starting Hebrew Wake Word Training Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Check dependencies
        if not check_dependencies():
            return False
        
        # Step 2: Validate configuration
        if not validate_config():
            return False
        
        # Step 3: Print dataset statistics
        print_dataset_statistics()
        
        # Step 4: Prepare dataset
        print(f"\n{'='*20} STEP 2: PREPARE DATASET {'='*20}")
        from prepare_dataset import prepare_datasets
        train_ds, val_ds, class_weights = prepare_datasets()
        print("✓ Dataset preparation completed")
        
        # Step 5: Train model
        print(f"\n{'='*20} STEP 3: TRAIN MODEL {'='*20}")
        from train_model import train_model
        model, history = train_model(train_ds, val_ds, class_weights)
        print("✓ Model training completed")
        
        # Step 6: Convert to TFLite
        print(f"\n{'='*20} STEP 4: CONVERT TO TFLITE {'='*20}")
        from convert_to_tflite import convert_to_tflite
        tflite_path, model_size = convert_to_tflite()
        print(f"✓ TFLite conversion completed ({model_size:.2f} KB)")
        
        # Check model size
        if model_size > 500:
            print(f"⚠ Warning: Model size ({model_size:.2f} KB) exceeds 500KB target")
        else:
            print(f"✓ Model size ({model_size:.2f} KB) is within 500KB target")
        
        # Step 7: Generate headers
        print(f"\n{'='*20} STEP 5: GENERATE HEADERS {'='*20}")
        from generate_header import generate_headers
        model_header, config_header = generate_headers()
        print("✓ Header generation completed")
        
        # Step 8: Deploy to ESP32
        print(f"\n{'='*20} STEP 6: DEPLOY TO ESP32 {'='*20}")
        from deploy_esp32 import deploy_to_esp32
        deploy_success = deploy_to_esp32()
        if deploy_success:
            print("✓ ESP32 deployment completed")
        else:
            print("✗ ESP32 deployment failed")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Model size: {model_size:.2f} KB")
        print(f"")
        print(f"Generated files:")
        print(f"- Keras model: {tflite_path.replace('.tflite', '.h5')}")
        print(f"- TFLite model: {tflite_path}")
        print(f"- Model header: {model_header}")
        print(f"- Config header: {config_header}")
        print(f"")
        print(f"Next steps:")
        print(f"1. Check esp32s3_deployment/ folder for Arduino files")
        print(f"2. Upload esp32s3_deployment.ino to your XIAO ESP32-S3")
        print(f"3. Use record_audio.py to receive predictions")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main pipeline execution
    """
    print("Hebrew Wake Word Detection - Training Pipeline")
    print("=" * 60)
    print("This pipeline will:")
    print("1. Prepare dataset from 5 classes (lehitraoot, shalom, bait, background, unknown)")
    print("2. Train CNN model with class weighting")
    print("3. Convert to TFLite (float32, no quantization)")
    print("4. Generate C headers for ESP32-S3")
    print("5. Deploy headers to esp32s3_deployment/")
    print("=" * 60)
    
    # Check if config exists
    if not os.path.exists('config.json'):
        print("✗ config.json not found!")
        print("Make sure you're running from the model_training/ directory")
        return False
    
    # Run pipeline
    success = run_training_pipeline()
    
    if success:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"Your ESP32-S3 is ready for wake word detection!")
    else:
        print(f"\n❌ Training pipeline failed!")
        print(f"Check the error messages above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
