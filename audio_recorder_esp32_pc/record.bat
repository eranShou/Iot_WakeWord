@echo off
REM Quick launcher for ESP32 Audio Recorder
echo ========================================
echo ESP32 Audio Recorder
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Check if pyserial is installed
python -c "import serial" >nul 2>&1
if errorlevel 1 (
    echo pyserial not found. Installing...
    pip install pyserial
    echo.
)

REM Run the recorder
python record_audio.py

pause



