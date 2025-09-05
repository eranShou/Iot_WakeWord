import wave
import numpy as np
import matplotlib.pyplot as plt

filename = "recording.wav"

with wave.open(filename, 'rb') as wf:
    frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)

plt.plot(samples[:2000])   # show first 2000 samples
plt.title("Waveform Preview")
plt.show()