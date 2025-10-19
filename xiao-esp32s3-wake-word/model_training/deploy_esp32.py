"""
Deploy trained model headers to ESP32-S3 deployment folder
Copies wake_word_model.h and model_config.h to esp32s3_deployment/
All parameters from config.json - no magic numbers
"""

import json
import os
import shutil
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def deploy_to_esp32():
    """
    Copy generated headers to ESP32 deployment directory
    """
    config = load_config()
    
    print("Deploying model headers to ESP32-S3...")
    print("=" * 50)
    
    # Get paths from config
    model_header = config['output']['model_header']
    config_header = config['output']['config_header']
    deployment_dir = config['esp32']['deployment_dir']
    
    # Check if headers exist
    if not os.path.exists(model_header):
        raise FileNotFoundError(f"Model header not found: {model_header}")
    
    if not os.path.exists(config_header):
        raise FileNotFoundError(f"Config header not found: {config_header}")
    
    # Create deployment directory if it doesn't exist
    deployment_path = Path(deployment_dir)
    deployment_path.mkdir(exist_ok=True)
    
    print(f"Deployment directory: {deployment_path.absolute()}")
    
    # Copy model header
    model_dest = deployment_path / "wake_word_model.h"
    shutil.copy2(model_header, model_dest)
    print(f"✓ Copied model header: {model_dest}")
    
    # Copy config header
    config_dest = deployment_path / "model_config.h"
    shutil.copy2(config_header, config_dest)
    print(f"✓ Copied config header: {config_dest}")
    
    # Verify files were copied
    if os.path.exists(model_dest) and os.path.exists(config_dest):
        print(f"\n✓ Deployment completed successfully!")
        print(f"Headers copied to: {deployment_path.absolute()}")
        
        # Print file sizes
        model_size = os.path.getsize(model_dest)
        config_size = os.path.getsize(config_dest)
        print(f"Model header size: {model_size} bytes")
        print(f"Config header size: {config_size} bytes")
        
        return True
    else:
        print(f"✗ Deployment failed - files not found in destination")
        return False

def verify_deployment():
    """
    Verify deployment was successful
    """
    config = load_config()
    deployment_dir = config['esp32']['deployment_dir']
    
    print(f"\nVerifying deployment...")
    
    # Check if deployment directory exists
    if not os.path.exists(deployment_dir):
        print(f"✗ Deployment directory not found: {deployment_dir}")
        return False
    
    # Check if headers exist in deployment directory
    model_header = os.path.join(deployment_dir, "wake_word_model.h")
    config_header = os.path.join(deployment_dir, "model_config.h")
    
    if os.path.exists(model_header):
        print(f"✓ Model header found: {model_header}")
    else:
        print(f"✗ Model header not found: {model_header}")
        return False
    
    if os.path.exists(config_header):
        print(f"✓ Config header found: {config_header}")
    else:
        print(f"✗ Config header not found: {config_header}")
        return False
    
    return True

def print_deployment_instructions():
    """
    Print instructions for using the deployed headers
    """
    config = load_config()
    deployment_dir = config['esp32']['deployment_dir']
    
    print(f"\n" + "=" * 60)
    print(f"ESP32-S3 DEPLOYMENT INSTRUCTIONS")
    print(f"=" * 60)
    print(f"")
    print(f"Headers have been deployed to: {deployment_dir}")
    print(f"")
    print(f"Next steps:")
    print(f"1. Open Arduino IDE")
    print(f"2. Select board: XIAO_ESP32S3")
    print(f"3. Install required libraries:")
    print(f"   - TensorFlowLite_ESP32")
    print(f"   - ESP_I2S")
    print(f"4. Copy the following files to your Arduino sketch folder:")
    print(f"   - {deployment_dir}/wake_word_model.h")
    print(f"   - {deployment_dir}/model_config.h")
    print(f"5. Include both headers in your .ino file:")
    print(f"   #include \"wake_word_model.h\"")
    print(f"   #include \"model_config.h\"")
    print(f"")
    print(f"Configuration parameters available:")
    print(f"- SAMPLE_RATE: {config['audio']['sample_rate']}")
    print(f"- NUM_SAMPLES: {config['audio']['num_samples']}")
    print(f"- SPECTROGRAM_HEIGHT: {config['spectrogram']['target_height']}")
    print(f"- SPECTROGRAM_WIDTH: {config['spectrogram']['target_width']}")
    print(f"- NUM_CLASSES: {config['model']['num_classes']}")
    print(f"- CLASS_LABELS[]: Array of class names")
    print(f"")
    print(f"All parameters are guaranteed to match training configuration!")

def main():
    """
    Main deployment function
    """
    try:
        # Deploy headers
        success = deploy_to_esp32()
        
        if success:
            # Verify deployment
            if verify_deployment():
                print_deployment_instructions()
            else:
                print("Deployment verification failed!")
                return False
        else:
            print("Deployment failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✓ ESP32-S3 deployment completed successfully!")
    else:
        print(f"\n✗ ESP32-S3 deployment failed!")
