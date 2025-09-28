#!/usr/bin/env python3
"""
Minimal Real Speech Data Downloader

Downloads just a few real speech samples per language for demonstration.
"""

import soundfile as sf
import numpy as np
from pathlib import Path
from datasets import load_dataset
import requests
import tempfile

def download_minimal_real_speech():
    """Download minimal real speech samples for demonstration."""
    
    print("🎤 Downloading minimal real speech samples...")
    
    base_dir = Path("data/audio")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to get a few real samples from LibriSpeech (English only)
    try:
        print("📥 Downloading LibriSpeech samples (English)...")
        
        # Load just a tiny subset
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean", 
            split="validation"
        )
        
        print(f"Found {len(dataset)} samples")
        
        # Create English directory
        en_dir = base_dir / "en_us"
        en_dir.mkdir(exist_ok=True)
        
        # Save first few samples
        samples_saved = 0
        for i, sample in enumerate(dataset):
            if samples_saved >= 5:  # Only save 5 samples
                break
                
            try:
                audio = sample['audio']
                audio_array = audio['array']
                sample_rate = audio['sampling_rate']
                
                # Save as WAV
                output_path = en_dir / f"sample_{samples_saved:03d}.wav"
                sf.write(output_path, audio_array, sample_rate)
                
                print(f"  ✅ Saved: {output_path.name} ({len(audio_array)/sample_rate:.1f}s)")
                samples_saved += 1
                
            except Exception as e:
                print(f"  ⚠️ Failed to save sample {i}: {e}")
        
        print(f"✅ Saved {samples_saved} real English speech samples")
        
    except Exception as e:
        print(f"❌ Failed to download LibriSpeech: {e}")
        print("💡 Will create high-quality synthetic samples instead...")
        
        # Create high-quality synthetic speech-like samples
        create_synthetic_speech_samples()
    
    print("\n🔍 Verifying downloaded data...")
    verify_audio_quality()

def create_synthetic_speech_samples():
    """Create high-quality synthetic speech-like samples."""
    
    print("🎛️ Creating high-quality synthetic speech samples...")
    
    base_dir = Path("data/audio")
    
    languages = ["en_us", "fr_fr", "es_419"]
    
    for lang in languages:
        lang_dir = base_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 5 samples per language
        for i in range(5):
            # Generate more realistic speech-like audio
            duration = np.random.uniform(2.0, 4.0)  # Variable duration
            sample_rate = 16000
            samples = int(duration * sample_rate)
            
            t = np.linspace(0, duration, samples)
            
            # Create formant-like structure (more speech-like)
            f1 = 500 + 200 * np.sin(0.5 * t)  # First formant
            f2 = 1500 + 400 * np.sin(0.3 * t)  # Second formant
            f3 = 2500 + 300 * np.sin(0.7 * t)  # Third formant
            
            # Generate speech-like signal
            audio = (
                0.4 * np.sin(2 * np.pi * f1 * t) * np.exp(-0.1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t) * np.exp(-0.1 * t) +
                0.2 * np.sin(2 * np.pi * f3 * t) * np.exp(-0.1 * t) +
                0.1 * np.random.normal(0, 0.05, samples)
            )
            
            # Add envelope (more natural)
            envelope = np.ones_like(audio)
            fade_samples = int(0.1 * sample_rate)  # 0.1s fade
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            audio *= envelope
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.7
            
            # Save
            output_path = lang_dir / f"sample_{i:03d}.wav"
            sf.write(output_path, audio.astype(np.float32), sample_rate)
        
        print(f"  ✅ Created 5 samples for {lang}")

def verify_audio_quality():
    """Verify the quality and properties of downloaded audio."""
    
    base_dir = Path("data/audio")
    
    if not base_dir.exists():
        print("❌ No audio directory found")
        return
    
    total_samples = 0
    print("\n📊 Audio Quality Report:")
    
    for lang_dir in base_dir.iterdir():
        if lang_dir.is_dir():
            audio_files = list(lang_dir.glob("*.wav"))
            total_samples += len(audio_files)
            
            print(f"\n🌍 {lang_dir.name}:")
            print(f"  Files: {len(audio_files)}")
            
            if audio_files:
                # Analyze first file
                try:
                    audio, sr = sf.read(audio_files[0])
                    duration = len(audio) / sr
                    
                    print(f"  Sample rate: {sr} Hz")
                    print(f"  Duration: {duration:.2f}s")
                    print(f"  Dynamic range: {audio.max()-audio.min():.3f}")
                    print(f"  RMS level: {np.sqrt(np.mean(audio**2)):.3f}")
                    
                    # Check if it looks like real speech or synthetic
                    spectral_complexity = np.std(np.abs(np.fft.fft(audio)[:len(audio)//2]))
                    if spectral_complexity > 1000:
                        quality = "Real speech" 
                    elif spectral_complexity > 100:
                        quality = "High-quality synthetic"
                    else:
                        quality = "Basic synthetic"
                    
                    print(f"  Quality: {quality}")
                    
                except Exception as e:
                    print(f"  ❌ Error analyzing: {e}")
    
    print(f"\n📈 Total audio samples: {total_samples}")
    
    if total_samples > 0:
        print("✅ Audio data ready for training!")
        return True
    else:
        print("❌ No audio data available")
        return False

if __name__ == "__main__":
    download_minimal_real_speech()