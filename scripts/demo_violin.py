#!/usr/bin/env python3
"""
Simple demo script for violin tuning.
This script generates violin tones and shows what the tuner should detect.
"""

import numpy as np
import sys
import os
import time

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import yin_pitch
from presets import freq_to_note

def generate_violin_tone(frequency, duration=3.0, sample_rate=44100, amplitude=0.4):
    """Generate a realistic violin tone with harmonics and vibrato."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Fundamental frequency
    fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics for violin-like sound
    harmonic2 = 0.25 * amplitude * np.sin(2 * np.pi * frequency * 2 * t)
    harmonic3 = 0.15 * amplitude * np.sin(2 * np.pi * frequency * 3 * t)
    harmonic4 = 0.08 * amplitude * np.sin(2 * np.pi * frequency * 4 * t)
    
    # Combine harmonics
    violin_tone = fundamental + harmonic2 + harmonic3 + harmonic4
    
    # Add vibrato (frequency modulation)
    vibrato_freq = 5.5  # 5.5 Hz vibrato
    vibrato_depth = 0.015  # 1.5% frequency modulation
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    
    violin_tone = violin_tone * vibrato
    
    # Add envelope (attack and decay)
    attack_samples = int(0.1 * sample_rate)  # 100ms attack
    decay_samples = int(0.2 * sample_rate)   # 200ms decay
    
    envelope = np.ones_like(t)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
    
    violin_tone = violin_tone * envelope
    
    return violin_tone.astype(np.float32)

def play_violin_string(note_name, frequency, duration=3.0):
    """Play a violin string and show detection results."""
    print(f"\n{'='*50}")
    print(f"Playing {note_name} ({frequency} Hz)")
    print('='*50)
    
    # Generate violin tone
    audio = generate_violin_tone(frequency, duration=duration)
    
    # Run pitch detection
    frames = yin_pitch(audio, fmin_hz=30.0, fmax_hz=1200.0)
    
    # Get detected frequencies
    detected_freqs = [f.f0_hz for f in frames if f.f0_hz > 0]
    
    if detected_freqs:
        avg_freq = np.mean(detected_freqs)
        detected_note, cents_off = freq_to_note(avg_freq)
        error_hz = abs(avg_freq - frequency)
        error_cents = abs(cents_off)
        
        print(f"Target: {note_name} ({frequency} Hz)")
        print(f"Detected: {detected_note} ({avg_freq:.2f} Hz)")
        print(f"Error: {error_hz:.2f} Hz ({error_cents:.1f} cents)")
        
        # Show tuning status
        if error_cents < 5:
            status = "PERFECT TUNE!"
        elif error_cents < 10:
            status = "Very close"
        elif error_cents < 25:
            status = "Good"
        elif error_cents < 50:
            status = "Needs tuning"
        else:
            status = "Way off"
        
        print(f"Status: {status}")
        
        if note_name in detected_note and error_cents < 50:
            print("Result: SUCCESS - Correctly detected!")
        else:
            print("Result: FAILED - Wrong note detected")
    else:
        print("Result: FAILED - No frequency detected")

def main():
    print("VIOLIN TUNER DEMO")
    print("="*50)
    print("This demo shows how the tuner detects violin strings.")
    print("Each string will be 'played' for 3 seconds.")
    print("="*50)
    
    # Violin strings
    violin_strings = [
        ("G3", 196.00, "Lowest string - G"),
        ("D4", 293.66, "Second string - D"), 
        ("A4", 440.00, "Third string - A (concert pitch)"),
        ("E5", 659.25, "Highest string - E"),
    ]
    
    for note_name, frequency, description in violin_strings:
        print(f"\n{description}")
        play_violin_string(note_name, frequency)
        time.sleep(0.5)  # Brief pause between strings
    
    print(f"\n{'='*50}")
    print("DEMO COMPLETE!")
    print("="*50)
    print("To use with real violin:")
    print("1. Start backend: python main.py")
    print("2. Start desktop app: python app.py") 
    print("3. Select 'violin_standard' preset")
    print("4. Use YIN algorithm (recommended)")
    print("5. Adjust VAD sensitivity if needed")
    print("="*50)

if __name__ == "__main__":
    main()
