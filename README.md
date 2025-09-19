## Wake Word Project by :  Eran Shouach, Yaron Florentz and Firas Gadban
  
## Details about the project
 Wake Word Optimization On Xiao ESP32S3

1. edge impulse model using controller mic
    * code for the recording using sd card is in `ESP32\record-to-sd\record-to-sd.ino` - this was write using GPT
    * code for the runnig of the model is in `ESP32\ei\ei.ino`
    * training and running of the model according to `https://www.youtube.com/watch?v=0X0vkzMOAA0` tutorial


## Folder description :
* ESP32: source code for the esp side (firmware).
* Documentation: wiring diagram + basic operating instructions
* Unit Tests: tests for individual hardware components (input / output devices)
* flutter_app : dart code for our Flutter app.
* Parameters: contains description of parameters and settings that can be modified IN YOUR CODE
* Assets: link to 3D printed parts, Audio files used in this project, Fritzing file for connection diagram (FZZ format) etc

## ESP32 SDK version used in this project: 
ei-sd: 2.0.16
mic_check,record-to-sd: 3.3.0


## Arduino/ESP32 libraries used in this project:
TBD

## Connection diagram:

## Project Poster:
 
This project is part of ICST - The Interdisciplinary Center for Smart Technologies, Taub Faculty of Computer Science, Technion
https://icst.cs.technion.ac.il/
