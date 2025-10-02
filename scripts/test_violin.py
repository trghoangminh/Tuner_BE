#!/usr/bin/env python3
"""
Test script for violin strings detection.
Violin tuning: G3 (196 Hz), D4 (293.66 Hz), A4 (440 Hz), E5 (659.25 Hz)
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import yin_pitch
from presets import freq_to_note, note_to_freq

def generate_tone(frequency, duration=2.0, sample_rate=44100, amplitude=0.5):
    """Generate a pure tone at the specified frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

def generate_violin_tone(frequency, duration=2.0, sample_rate=44100, amplitude=0.5):
    """Generate a more realistic violin tone with harmonics."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Fundamental frequency
    fundamental = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonics for more realistic violin sound
    # 2nd harmonic (octave)
    harmonic2 = 0.3 * amplitude * np.sin(2 * np.pi * frequency * 2 * t)
    # 3rd harmonic (perfect fifth)
    harmonic3 = 0.2 * amplitude * np.sin(2 * np.pi * frequency * 3 * t)
    # 4th harmonic (double octave)
    harmonic4 = 0.1 * amplitude * np.sin(2 * np.pi * frequency * 4 * t)
    
    # Combine all harmonics
    violin_tone = fundamental + harmonic2 + harmonic3 + harmonic4
    
    # Add some vibrato (frequency modulation)
    vibrato_freq = 5.0  # 5 Hz vibrato
    vibrato_depth = 0.02  # 2% frequency modulation
    vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    
    violin_tone = violin_tone * vibrato
    
    return violin_tone.astype(np.float32)

def test_violin_strings():
    """Test all violin strings detection."""
    print("Violin Strings Detection Test")
    print("="*50)
    print("Violin tuning: G3, D4, A4, E5")
    print("="*50)
    
    # Violin strings with their frequencies
    violin_strings = [
        ("G3", 196.00, "Lowest string - G"),
        ("D4", 293.66, "Second string - D"), 
        ("A4", 440.00, "Third string - A (concert pitch)"),
        ("E5", 659.25, "Highest string - E"),
    ]
    
    # Only YIN algorithm available now
    algo = "yin"
    print(f"\n{'='*60}")
    print(f"Testing with {algo.upper()} algorithm")
    print('='*60)
        
    for note_name, target_freq, description in violin_strings:
        print(f"\n--- {note_name} ({target_freq} Hz) - {description} ---")
        
        # Generate violin tone
        audio = generate_violin_tone(target_freq, duration=2.0, amplitude=0.3)
        
        # Run pitch detection with YIN algorithm
        frames = yin_pitch(audio, fmin_hz=30.0, fmax_hz=1200.0)
            
        # Get detected frequencies
        detected_freqs = [f.f0_hz for f in frames if f.f0_hz > 0]
        
        if detected_freqs:
            avg_freq = np.mean(detected_freqs)
            detected_note, cents_off = freq_to_note(avg_freq)
            error_hz = abs(avg_freq - target_freq)
            error_cents = abs(cents_off)
            
            print(f"  Target: {note_name} ({target_freq} Hz)")
            print(f"  Detected: {detected_note} ({avg_freq:.2f} Hz)")
            print(f"  Error: {error_hz:.2f} Hz ({error_cents:.1f} cents)")
            print(f"  Valid detections: {len(detected_freqs)}/{len(frames)}")
            
            # Check if it's correct
            if note_name in detected_note and error_cents < 50:  # Within 50 cents
                print(f"  Result: PASS")
            else:
                print(f"  Result: FAIL - Expected {note_name}, got {detected_note}")
        else:
            print(f"  Result: FAIL - No frequency detected")

def test_violin_with_vibrato():
    """Test violin detection with vibrato (more realistic)."""
    print(f"\n{'='*60}")
    print("Testing Violin with Vibrato (Realistic)")
    print('='*60)
    
    # Test A4 with different vibrato rates
    vibrato_rates = [4.0, 5.0, 6.0, 7.0]  # Hz
    
    for vibrato_rate in vibrato_rates:
        print(f"\n--- A4 with {vibrato_rate} Hz vibrato ---")
        
        # Generate A4 with specific vibrato rate
        t = np.linspace(0, 2.0, int(44100 * 2.0), False)
        fundamental = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        vibrato = 1.0 + 0.03 * np.sin(2 * np.pi * vibrato_rate * t)
        audio = (fundamental * vibrato).astype(np.float32)
        
        # Test with YIN (better for vibrato)
        frames = yin_pitch(audio, fmin_hz=30.0, fmax_hz=1200.0)
        detected_freqs = [f.f0_hz for f in frames if f.f0_hz > 0]
        
        if detected_freqs:
            avg_freq = np.mean(detected_freqs)
            detected_note, cents_off = freq_to_note(avg_freq)
            error_hz = abs(avg_freq - 440.0)
            error_cents = abs(cents_off)
            
            print(f"  Detected: {detected_note} ({avg_freq:.2f} Hz)")
            print(f"  Error: {error_hz:.2f} Hz ({error_cents:.1f} cents)")
            
            if "A4" in detected_note and error_cents < 50:
                print(f"  Result: PASS")
            else:
                print(f"  Result: FAIL")

def test_violin_octave_jumps():
    """Test violin detection across different octaves."""
    print(f"\n{'='*60}")
    print("Testing Violin Octave Range")
    print('='*60)
    
    # Test frequencies around violin range
    test_freqs = [
        (196.0, "G3"),
        (220.0, "A3"), 
        (246.94, "B3"),
        (293.66, "D4"),
        (329.63, "E4"),
        (440.0, "A4"),
        (493.88, "B4"),
        (523.25, "C5"),
        (587.33, "D5"),
        (659.25, "E5"),
        (698.46, "F5"),
        (783.99, "G5"),
    ]
    
    print("Testing YIN algorithm:")
    for freq, expected_note in test_freqs:
        audio = generate_violin_tone(freq, duration=1.0, amplitude=0.3)
        frames = yin_pitch(audio, fmin_hz=30.0, fmax_hz=1200.0)
        detected_freqs = [f.f0_hz for f in frames if f.f0_hz > 0]
        
        if detected_freqs:
            avg_freq = np.mean(detected_freqs)
            detected_note, cents_off = freq_to_note(avg_freq)
            error_cents = abs(cents_off)
            
            status = "PASS" if expected_note in detected_note and error_cents < 50 else "FAIL"
            print(f"  {freq:6.1f} Hz -> {detected_note:3s} ({avg_freq:6.1f} Hz) {error_cents:4.1f}c {status}")

def main():
    print("VIOLIN TUNER TEST")
    print("="*50)
    
    test_violin_strings()
    test_violin_with_vibrato()
    test_violin_octave_jumps()
    
    print(f"\n{'='*60}")
    print("DEMO INSTRUCTIONS:")
    print("="*60)
    print("1. Start the backend server: python main.py")
    print("2. Start the desktop app: python app.py")
    print("3. Select 'violin_standard' preset")
    print("4. Test each string:")
    print("   - G3 (196 Hz) - Lowest string")
    print("   - D4 (293.66 Hz) - Second string") 
    print("   - A4 (440 Hz) - Third string (concert pitch)")
    print("   - E5 (659.25 Hz) - Highest string")
    print("5. Try different algorithms (YIN recommended)")
    print("6. Adjust VAD sensitivity if needed")
    print("="*60)

if __name__ == "__main__":
    main()
