"""
Custom STFT Implementation
Identical FFT algorithm for both Python training and ESP32 C++ deployment
Ensures perfect alignment between training and inference
"""

import numpy as np
import math

def custom_fft(data):
    """
    Radix-2 FFT implementation in pure Python/NumPy
    Matches C++ implementation exactly
    """
    n = len(data)
    if n == 1:
        return data
    
    # Ensure power of 2
    if n & (n - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        padded_data = np.zeros(next_pow2, dtype=complex)
        padded_data[:n] = data
        data = padded_data
        n = next_pow2
    
    # Bit-reversal permutation
    result = np.zeros(n, dtype=complex)
    for i in range(n):
        j = 0
        temp = i
        for k in range(int(math.log2(n))):
            j = (j << 1) | (temp & 1)
            temp >>= 1
        result[j] = data[i]
    
    # FFT computation (Cooley-Tukey)
    for length in [2**i for i in range(1, int(math.log2(n)) + 1)]:
        wlen = -2 * math.pi / length
        for i in range(0, n, length):
            w = 1.0
            for j in range(length // 2):
                u = result[i + j]
                v = result[i + j + length // 2] * complex(math.cos(wlen * j), math.sin(wlen * j))
                result[i + j] = u + v
                result[i + j + length // 2] = u - v
    
    return result

def apply_hann_window(frame, frame_length):
    """
    Apply Hann window to frame
    Matches C++ implementation exactly
    """
    hann_window = np.zeros(frame_length)
    for i in range(frame_length):
        hann_window[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (frame_length - 1)))
    
    return frame * hann_window

def compute_stft_custom(audio, frame_length, frame_step, fft_length, output_height, output_width):
    """
    Custom STFT implementation matching ESP32 C++ exactly
    Computes 32x32 spectrogram directly without resize
    
    Args:
        audio: 1D numpy array of audio samples
        frame_length: Length of each frame (255)
        frame_step: Step size between frames (128)
        fft_length: FFT length (256)
        output_height: Output spectrogram height (32)
        output_width: Output spectrogram width (32)
    
    Returns:
        2D numpy array of shape (output_height, output_width)
    """
    # Calculate number of frames exactly as in C++ implementation
    num_frames = min((len(audio) - frame_length) // frame_step + 1, output_height)
    
    # Initialize output spectrogram
    spectrogram = np.zeros((output_height, output_width), dtype=np.float32)
    
    # Process each frame
    for frame in range(num_frames):
        frame_start = frame * frame_step
        
        # Skip if we don't have enough samples
        if frame_start + frame_length > len(audio):
            break
        
        # Extract frame
        frame_data = audio[frame_start:frame_start + frame_length].astype(np.float32)
        
        # Apply Hann window
        windowed_frame = apply_hann_window(frame_data, frame_length)
        
        # Pad frame to FFT length if needed
        if len(windowed_frame) < fft_length:
            padded_frame = np.zeros(fft_length, dtype=np.float32)
            padded_frame[:len(windowed_frame)] = windowed_frame
            windowed_frame = padded_frame
        elif len(windowed_frame) > fft_length:
            windowed_frame = windowed_frame[:fft_length]
        
        # Convert to complex for FFT
        complex_frame = windowed_frame.astype(np.complex64)
        
        # Compute FFT
        fft_result = custom_fft(complex_frame)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_result)
        
        # Store magnitude spectrum in output spectrogram
        # Only use first half of FFT result (positive frequencies)
        num_bins = min(fft_length // 2, output_width)
        for i in range(num_bins):
            spectrogram[frame, i] = magnitude[i]
    
    return spectrogram

def validate_stft_parameters(audio_length, frame_length, frame_step, fft_length, output_height, output_width):
    """
    Validate STFT parameters and print debug information
    """
    print(f"STFT Parameters:")
    print(f"  Audio length: {audio_length}")
    print(f"  Frame length: {frame_length}")
    print(f"  Frame step: {frame_step}")
    print(f"  FFT length: {fft_length}")
    print(f"  Output size: {output_height}x{output_width}")
    
    # Calculate number of possible frames
    max_frames = (audio_length - frame_length) // frame_step + 1
    print(f"  Max possible frames: {max_frames}")
    print(f"  Using first {min(max_frames, output_height)} frames")
    
    # Calculate frequency bins
    freq_bins = fft_length // 2
    print(f"  FFT frequency bins: {freq_bins}")
    print(f"  Using first {min(freq_bins, output_width)} bins")
    
    return max_frames, freq_bins
