# XIAO ESP32S3 Audio Recorder - Quick Start

**Super simple guide to record audio with XIAO ESP32S3 built-in microphone!** 🎤

## 🛒 What You Need

- ✅ Seeed Studio XIAO ESP32S3 (with built-in microphone)
- ✅ USB-C cable

**That's ALL!** No external microphone, no wiring, no SD card! 🎉

---

## 🎤 Built-in Microphone

The XIAO ESP32S3 has a built-in PDM microphone (MSM261D3526H1CPM).

**NO WIRING NEEDED!** Just connect USB and upload the sketch.

The microphone hole is on the side of the board. Speak towards it for best results.

---

## 💻 Software Setup (5 minutes)

### 1. Install Python Library

```bash
pip install pyserial
```

### 2. Upload Arduino Sketch

1. **Open Arduino IDE**
2. **Open** `audio_recorder_esp32_pc.ino`
3. **Select Board:**
   - Tools → Board → ESP32 Arduino → **XIAO_ESP32S3**
4. **Select Port:**
   - Tools → Port → (your COM port)
5. **Upload** (click → arrow button)

**That's it!** The XIAO uploads automatically, no BOOT button needed!

---

## 🎙️ Record Audio (2 minutes)

```bash
cd audio_recorder_esp32_pc
python record_audio.py
```

### What happens:
1. Script finds your XIAO automatically
2. Press Enter (or it starts automatically)
3. Records 5 seconds
4. Saves to `recordings/recording_TIMESTAMP.wav`
5. Ask if you want to record more (y/n)

---

## ✅ Test It Works

After upload, run:

```bash
python test_connection.py
```

You should see:
```
✓ ESP32 is connected and responding
✓ I2S_OK
✓ WAITING
```

If yes → **You're ready!** 🎉

---

## 🎯 For Wake Word Training

Perfect workflow:

```bash
# 1. Record "Shalom" samples
python record_audio.py
# (say "Shalom" 5 times, press 'y' between recordings)

# 2. Move to training folder
mkdir -p ../data/raw/Shalom
mv recordings/*.wav ../data/raw/Shalom/

# 3. Record "Lehitraot" samples
python record_audio.py
# (say "Lehitraot" 5 times)

mkdir -p ../data/raw/Lehitraot
mv recordings/*.wav ../data/raw/Lehitraot/

# 4. Record background noise
python record_audio.py
# (silence, ambient noise, random words)

mkdir -p ../data/raw/Background
mv recordings/*.wav ../data/raw/Background/

# 5. Train model
cd ..
python scripts/preprocess.py
python scripts/train_model.py
```

---

## 🐛 Troubleshooting

### "Port in use" error
- ❌ Close Arduino Serial Monitor
- ✅ Run script again

### No audio / silent recording
- ❌ Make sure you're speaking TOWARDS the mic hole on the board
- ❌ Built-in mic is less sensitive - speak **louder** and **closer**
- ✅ Try clapping or tapping near the board during recording
- ✅ Make sure you selected **XIAO_ESP32S3** board in Arduino IDE

### "ESP32 not responding"
- ❌ Make sure sketch is uploaded
- ✅ Press reset button on XIAO
- ✅ Run `python test_connection.py`

### Corrupted audio
- Try lower baud rate:
  - Change `Serial.begin(921600)` → `Serial.begin(460800)` in sketch
  - Change `BAUD_RATE = 921600` → `BAUD_RATE = 460800` in Python

---

## 📝 Quick Reference

**Hardware:**
- XIAO ESP32S3 with built-in microphone
- Microphone hole is on the side of the board
- No wiring needed!

**Commands:**
```bash
pip install pyserial           # Install
python record_audio.py         # Record
python test_connection.py      # Diagnose
```

**Files:**
- `audio_recorder_esp32_pc.ino` - Arduino sketch
- `record_audio.py` - Recording script
- `test_connection.py` - Diagnostic tool
- `recordings/` - Your recordings appear here

---

## 🎓 Tips

1. **Speak 20-30cm from microphone** for best results
2. **Record in quiet room** - close windows, turn off fans
3. **Collect 10-20 samples** per wake word minimum
4. **Include variety** - different tones, speeds, volumes
5. **Background samples** - TV, music, other voices

---

## 🚀 That's It!

You're ready to collect training data for your wake word model!

**Total time:** ~12 minutes from zero to recording! 🎉

For detailed info, see `README.md`

