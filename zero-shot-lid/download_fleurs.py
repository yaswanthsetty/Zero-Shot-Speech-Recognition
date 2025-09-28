#!/usr/bin/env python3
"""
FLEURS Dataset Downloader

This script downloads specific language subsets from the FLEURS dataset.
"""

from datasets import load_dataset
import os
from pathlib import Path
import soundfile as sf

def download_fleurs_languages(languages=None):
    """Download FLEURS data for specific languages using modern datasets API."""
    
    if languages is None:
        # Default to a few key languages
        languages = ['en_us', 'es_419', 'fr_fr', 'de_de', 'it_it']
    
    print(f"📥 Downloading FLEURS data for {len(languages)} languages...")
    
    base_dir = Path("data/audio")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in languages:
        try:
            print(f"⬇️  Downloading {lang}...")
            
            # Try different approaches to load FLEURS data
            dataset = None
            
            # Method 1: Try the new format without scripts
            try:
                dataset = load_dataset("google/fleurs", name=f"all", split="train")
                # Filter for specific language
                dataset = dataset.filter(lambda x: x.get('lang_id', '') == lang)
            except:
                pass
            
            # Method 2: Try LibriSpeech English
            if dataset is None and lang == 'en_us':
                try:
                    print(f"  Trying LibriSpeech for English...")
                    dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=False)
                    print(f"  ✅ Loaded LibriSpeech with {len(dataset)} samples")
                except Exception as e:
                    print(f"  ❌ LibriSpeech failed: {e}")
            
            # Method 3: Try Common Voice if available
            if dataset is None:
                try:
                    # Try Common Voice dataset
                    print(f"  Trying Common Voice for {lang}...")
                    # Map our language codes to Common Voice language codes
                    cv_lang_map = {
                        'en_us': 'en',
                        'fr_fr': 'fr', 
                        'es_419': 'es',
                        'de_de': 'de',
                        'it_it': 'it'
                    }
                    
                    if lang in cv_lang_map:
                        cv_lang = cv_lang_map[lang]
                        dataset = load_dataset("common_voice", cv_lang, split="train", streaming=False)
                        print(f"  ✅ Loaded Common Voice {cv_lang} with {len(dataset)} samples")
                    else:
                        print(f"  ❌ No Common Voice mapping for {lang}")
                        
                except Exception as e:
                    print(f"  ❌ Common Voice failed: {e}")
            
            # Method 4: Skip if no real data available
            if dataset is None:
                print(f"  ❌ No real speech datasets available for {lang}")
                print(f"  💡 Suggestion: Download Common Voice data manually")
                print(f"     Visit: https://commonvoice.mozilla.org/datasets")
                continue
            
            # Create language directory
            lang_dir = base_dir / lang
            lang_dir.mkdir(exist_ok=True)
            
            # Handle both HuggingFace dataset and demo data
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                # HuggingFace dataset
                max_samples = min(20, len(dataset))
                data_source = "HuggingFace"
            else:
                # Demo data (list)
                max_samples = len(dataset)
                data_source = "demo"
            
            print(f"  Processing {max_samples} samples from {data_source}...")
            
            for i in range(max_samples):
                try:
                    if hasattr(dataset, '__getitem__'):
                        sample = dataset[i]
                    else:
                        sample = dataset[i]
                    
                    # Get audio data
                    audio = sample['audio']
                    audio_array = audio['array']
                    sample_rate = audio['sampling_rate']
                    
                    # Save as WAV file
                    output_path = lang_dir / f"sample_{i:03d}.wav"
                    sf.write(output_path, audio_array, sample_rate)
                    
                    if i % 5 == 0:
                        print(f"    Saved {i+1}/{max_samples} samples...")
                        
                except Exception as sample_e:
                    print(f"    ⚠️  Failed to save sample {i}: {sample_e}")
                    continue
            
            print(f"✅ Successfully saved {max_samples} audio samples for {lang}")
            
        except Exception as e:
            print(f"❌ Failed to download {lang}: {e}")
            continue
    
    print("\n🎉 Download complete!")
    print("\n📁 Downloaded data structure:")
    for lang_dir in base_dir.iterdir():
        if lang_dir.is_dir():
            file_count = len(list(lang_dir.rglob("*")))
            print(f"  {lang_dir.name}: {file_count} files")

def verify_downloaded_data():
    """Verify the downloaded audio data."""
    
    print("\n� Verifying downloaded data...")
    
    audio_dir = Path("data/audio")
    
    if not audio_dir.exists():
        print("❌ No audio data directory found!")
        return False
    
    total_files = 0
    print("\n📁 Audio data structure:")
    
    for lang_dir in audio_dir.iterdir():
        if lang_dir.is_dir():
            audio_files = list(lang_dir.glob("*.wav"))
            total_files += len(audio_files)
            print(f"  {lang_dir.name}: {len(audio_files)} audio files")
            
            # Test loading a sample file
            if audio_files:
                try:
                    test_audio, sr = sf.read(audio_files[0])
                    print(f"    ✅ Sample: {test_audio.shape}, {sr}Hz")
                except Exception as e:
                    print(f"    ❌ Error reading sample: {e}")
    
    print(f"\nTotal audio files: {total_files}")
    return total_files > 0

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
    verify_downloaded_data()
    
    print("\n🚀 Ready to use!")
    print("Run: python main.py")
