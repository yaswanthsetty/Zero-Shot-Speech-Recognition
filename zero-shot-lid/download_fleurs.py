#!/usr/bin/env python3
"""
FLEURS Dataset Downloader

This script downloads specific language subsets from the FLEURS dataset.
"""

from huggingface_hub import snapshot_download
import os
from pathlib import Path

def download_fleurs_languages(languages=None):
    """Download FLEURS data for specific languages."""
    
    if languages is None:
        # Default to a few key languages
        languages = ['en_us', 'es_419', 'fr_fr', 'de_de', 'it_it']
    
    print(f"📥 Downloading FLEURS data for {len(languages)} languages...")
    
    base_dir = Path("data/fleurs_raw")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in languages:
        try:
            print(f"⬇️  Downloading {lang}...")
            
            # Download to language-specific directory
            lang_dir = base_dir / lang
            
            snapshot_download(
                repo_id="google/fleurs",
                repo_type="dataset",
                local_dir=str(lang_dir),
                allow_patterns=f"data/{lang}/*",  # Only download this language
                cache_dir=".cache/huggingface"
            )
            
            print(f"✅ Successfully downloaded {lang}")
            
        except Exception as e:
            print(f"❌ Failed to download {lang}: {e}")
            continue
    
    print("\n🎉 Download complete!")
    print("\n📁 Downloaded data structure:")
    for lang_dir in base_dir.iterdir():
        if lang_dir.is_dir():
            file_count = len(list(lang_dir.rglob("*")))
            print(f"  {lang_dir.name}: {file_count} files")

def convert_to_local_format():
    """Convert downloaded FLEURS data to local audio format."""
    
    print("\n🔄 Converting to local audio format...")
    
    import pandas as pd
    from datasets import load_dataset
    import soundfile as sf
    import shutil
    
    fleurs_dir = Path("data/fleurs_raw")
    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each language directory
    for lang_dir in fleurs_dir.iterdir():
        if not lang_dir.is_dir():
            continue
            
        lang_code = lang_dir.name
        output_lang_dir = audio_dir / lang_code
        output_lang_dir.mkdir(exist_ok=True)
        
        try:
            print(f"Processing {lang_code}...")
            
            # Look for audio files in the downloaded data
            audio_files = list(lang_dir.rglob("*.wav")) + list(lang_dir.rglob("*.mp3"))
            
            if audio_files:
                # Copy audio files with sequential naming
                for i, audio_file in enumerate(audio_files[:50]):  # Limit to 50 files
                    output_file = output_lang_dir / f"sample_{i:03d}.wav"
                    
                    # Copy and potentially convert
                    if audio_file.suffix == '.wav':
                        shutil.copy2(audio_file, output_file)
                    else:
                        # Convert other formats to WAV using soundfile
                        try:
                            audio, sr = sf.read(audio_file)
                            sf.write(output_file, audio, sr)
                        except Exception as convert_e:
                            print(f"  ⚠️  Failed to convert {audio_file}: {convert_e}")
                            continue
                
                print(f"  ✅ Converted {min(len(audio_files), 50)} files for {lang_code}")
            else:
                print(f"  ⚠️  No audio files found for {lang_code}")
                
        except Exception as e:
            print(f"  ❌ Error processing {lang_code}: {e}")
            continue
    
    print("\n🎉 Conversion complete!")
    print("\n📁 Final audio structure:")
    for lang_dir in audio_dir.iterdir():
        if lang_dir.is_dir():
            audio_files = list(lang_dir.glob("*.wav"))
            print(f"  {lang_dir.name}: {len(audio_files)} audio files")

if __name__ == "__main__":
    # List of languages to download (you can modify this)
    languages_to_download = [
        'en_us',    # English
        'es_419',   # Spanish
        'fr_fr',    # French
        'de_de',    # German
        'it_it',    # Italian
        # Add more languages as needed from the SEEN_LANGUAGES list
    ]
    
    download_fleurs_languages(languages_to_download)
    convert_to_local_format()
    
    print("\n🚀 Ready to use!")
    print("Run: python main.py")
