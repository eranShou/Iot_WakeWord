## Wake Word Project by :  Eran Shouach, Yaron Florentz and Firas Gadban
  
## Details about the project
 Wake Word Optimization On Xiao ESP32S3
1. ### data collection:
    #### using sd card:
     1. code in `ESP32\record-to-sd\record-to-sd.ino`
     2.  flush on your chip - make sure microphone pins are correct in `setPinsPdmRx` and that the sd card is connected ( it will show error message in serial if not)
     3. to record enter recording file name to serial and press 'Enter' then type `rec` to start recordin ( while recording light is on and it will turn off when finished)
     4. notes: you can change recording length in the `LEN` variable and sample rate in the `SAMPLERATE` variable
    
    #### using phone
    part of the data collection we use whatapp audio (translate to .wav files) to create our own data set - all the data is in the drive

    #### using ivrit.ai
    ivrit.ai is an open source project for generation of Hebrew speech samples through processing and segmentation of transcribed podcast recordings `https://www.ivrit.ai/`
    1. code to the API to generate dataset in hebrew - `ivrit-ai dataset/IvritAiDataFetch.py`
    2. usage: 


### 2. edge impulse model
1. training and running of the model according to `https://www.youtube.com/watch?v=0X0vkzMOAA0` tutorial
2. code for the runnig of the model is in `ESP32\ei\ei.ino`, import (via sketch -> include library -> add .ZIP library in Arduino IDE) the egde impulse deploy library and replace the include with our to use your own model
3. the code will run and show the results of the model on the serial, it takes a one second recording and classify it using included model. the results of the classification is saved in the variable `result.classification` where the label is saved under `result.classification[index].label` and the value is saved under `result.classification[index].value`. using the result is recomended via those fields.
4. notes and ways to improve preformance:
    1. training without unknown key gives false positive - meaning every word that not sound like noise is classefied to one of the labels

### 3. TF Lite Micro - MFCC
1. code in `ESP32/tflm/tflm.ino`  for controller and in `TFLM/train_mfcc.py` to train model.
2. to train model change labels number and names in the `train_mfcc.py` script, enter the data to the data folder by label (a folder to each label), run the script and then run `xxd -i TFLM/wakeword_model_esp32mfcc.tflite > ESP32/tflm/model_data.h` (on linux)
3. flush the code in the tflm directory to your controller
4. notes - the mfcc is basic and less accurate than other options


## Folder description :
* ESP32: source code for the esp side (firmware).
* Documentation: wiring diagram + basic operating instructions
* Unit Tests: tests for individual hardware components (input / output devices)
* flutter_app : dart code for our Flutter app.
* Parameters: contains description of parameters and settings that can be modified IN YOUR CODE
* Assets: link to 3D printed parts, Audio files used in this project, Fritzing file for connection diagram (FZZ format) etc

## ESP32 SDK version used in this project: 
ei: 2.0.16
record to sd: 3.3.0
tensorflow Lite Micro MFCC & STFT: 2.0.16  


## Arduino/ESP32 libraries used in this project:
for record to sd:
* ESP_I2S.h ( part of the 3.3.0 ESP32 SDK)
* FS.h (part of the 3.3.0 ESP32 SDK)
* SD.h (part of arduino core)

for edge impulse:
* I2S.h ( part of the 2.0.16 ESP32 SDK)
* ei costume library

TensorFlow Lite Micro MFCC:
* I2S.h ( part of the 2.0.16 ESP32 SDK)
* arduinoFFT.h v2.0.4
* math.h (part of arduino core)
* tflm_esp32.h v2.0.0
* eloquent_tinyml.h v3.0.1



## Connection diagram:

## Project Poster:
 
This project is part of ICST - The Interdisciplinary Center for Smart Technologies, Taub Faculty of Computer Science, Technion
https://icst.cs.technion.ac.il/
