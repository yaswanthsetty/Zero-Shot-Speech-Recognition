#!/usr/bin/env python3
"""
Common Voice to FLEURS Format Converter

Usage:
    python convert_common_voice.py /path/to/common_voice_lang.tar.gz lang_code
"""

import sys
import os
import tarfile
import pandas as pd
import soundfile as sf
import shutil
from pathlib import Path

def convert_common_voice(tar_path, lang_code, max_samples=100):
    """Convert Common Voice data to our format."""
    
    output_dir = Path(f"data/audio/{lang_code}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting Common Voice data for {lang_code}...")
    
    # Extract tar file
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall("temp_cv")
    
    # Find the clips directory
    clips_dir = None
    for root, dirs, files in os.walk("temp_cv"):
        if "clips" in dirs:
            clips_dir = Path(root) / "clips"
            break
    
    if not clips_dir:
        print("❌ Could not find clips directory")
        return
    
    # Copy audio files with proper naming
    audio_files = list(clips_dir.glob("*.mp3"))[:max_samples]
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load and resample to 16kHz
            audio, sr = sf.read(audio_file)
            if sr != 16000:
                # Simple resampling (for proper resampling, use librosa)
                audio = audio[::sr//16000] if sr > 16000 else audio
            
            # Save as wav
            output_path = output_dir / f"sample_{i:03d}.wav"
            sf.write(output_path, audio, 16000)
            
            if i % 10 == 0:
                print(f"  Converted {i+1}/{len(audio_files)} files...")
                
        except Exception as e:
            print(f"  ❌ Failed to convert {audio_file}: {e}")
    
    # Cleanup
    shutil.rmtree("temp_cv")
    print(f"✅ Converted {len(audio_files)} files for {lang_code}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_common_voice.py <tar_path> <lang_code>")
        sys.exit(1)
    
    convert_common_voice(sys.argv[1], sys.argv[2])
