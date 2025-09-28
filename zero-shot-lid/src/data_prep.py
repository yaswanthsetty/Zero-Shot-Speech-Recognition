"""
Data preparation module for Zero-Shot Spoken Language Identification.

This module handles loading and splitting the FLEURS dataset for training
and evaluation purposes. Now includes support for local audio files.
"""

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import random
from pathlib import Path
try:
    import soundfile as sf
except ImportError:
    sf = None
from . import config


def load_local_audio_data(data_dir="data/audio"):
    """
    Load audio data from local directory structure.
    
    Expected structure:
    data/audio/
    ├── en_us/
    │   ├── sample_001.wav
    │   └── sample_002.wav
    ├── es_419/
    │   └── sample_001.wav
    └── ...
    
    Returns:
        Dict mapping language codes to lists of audio data items
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"❌ Local audio directory {data_dir} does not exist")
        return {}
    
    if sf is None:
        print("❌ soundfile not installed. Install with: pip install soundfile")
        return {}
    
    datasets_by_lang = {}
    
    for lang_dir in data_dir.iterdir():
        if lang_dir.is_dir():
            lang_code = lang_dir.name
            audio_files = list(lang_dir.glob("*.wav")) + list(lang_dir.glob("*.mp3")) + list(lang_dir.glob("*.flac"))
            
            if audio_files:
                lang_data = []
                for audio_file in audio_files:
                    try:
                        # Load audio file
                        audio_array, sample_rate = sf.read(audio_file)
                        
                        # Ensure mono audio
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.mean(axis=1)
                        
                        # Resample to 16kHz if needed (simple approach)
                        if sample_rate != 16000:
                            # Simple resampling - for better quality, use librosa
                            if sample_rate > 16000:
                                step = sample_rate // 16000
                                audio_array = audio_array[::step]
                            else:
                                # Upsample by repetition (crude but works)
                                repeat_factor = 16000 // sample_rate
                                audio_array = np.repeat(audio_array, repeat_factor)
                            sample_rate = 16000
                        
                        # Create data item compatible with FLEURS format
                        item = {
                            'audio': {
                                'array': audio_array.astype(np.float32),
                                'sampling_rate': sample_rate
                            },
                            'lang_id': lang_code,
                            'id': f"{lang_code}_{audio_file.stem}"
                        }
                        lang_data.append(item)
                        
                    except Exception as e:
                        print(f"❌ Failed to load {audio_file}: {e}")
                
                if lang_data:
                    datasets_by_lang[lang_code] = lang_data
                    print(f"✅ Loaded {len(lang_data)} local audio samples for {lang_code}")
    
    return datasets_by_lang


def create_synthetic_dataset(seen_langs: List[str], unseen_langs: List[str]):
    """
    Create synthetic audio datasets for demonstration purposes.
    
    This function creates mock audio data when the real FLEURS dataset
    cannot be loaded. It generates random audio arrays that simulate
    speech recordings.
    
    Args:
        seen_langs: List of seen language codes
        unseen_langs: List of unseen language codes
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    print("Creating synthetic audio data...")
    
    # Audio parameters
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    samples_per_audio = int(sample_rate * duration)
    
    # Create training data (seen languages only)
    train_data = []
    val_data = []
    
    samples_per_lang = 20  # Small number for demo
    
    # Create mock dataset objects  
    class MockDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __iter__(self):
            return iter(self.data)
    
    # Generate data for seen languages (training/validation)
    for lang in seen_langs[:5]:  # Limit to 5 languages for demo
        for i in range(samples_per_lang):
            audio_signal = generate_synthetic_audio(lang, duration, sample_rate, samples_per_audio)
            
            audio_data = {
                'array': audio_signal.astype(np.float32),
                'sampling_rate': sample_rate
            }
            
            item = {
                'audio': audio_data,
                'lang_id': lang,
                'id': f"synthetic_{lang}_{i}"
            }
            
            # 80% train, 20% validation
            if i < samples_per_lang * 0.8:
                train_data.append(item)
            else:
                val_data.append(item)
    
    # Generate test data for unseen languages
    test_data = []
    for lang in unseen_langs[:3]:  # Limit to 3 unseen languages for demo
        for i in range(samples_per_lang // 2):  # Fewer test samples
            audio_signal = generate_synthetic_audio(lang, duration, sample_rate, samples_per_audio)
            
            audio_data = {
                'array': audio_signal.astype(np.float32),
                'sampling_rate': sample_rate
            }
            
            item = {
                'audio': audio_data,
                'lang_id': lang,
                'id': f"synthetic_{lang}_{i}"
            }
            
            test_data.append(item)
    
    print(f"Created synthetic dataset:")
    print(f"  Training samples: {len(train_data)} (seen languages)")
    print(f"  Validation samples: {len(val_data)} (seen languages)")
    print(f"  Test samples: {len(test_data)} (unseen languages)")
    print(f"  Seen languages: {seen_langs[:5]}")
    print(f"  Unseen languages: {unseen_langs[:3]}")
    
    return MockDataset(train_data), MockDataset(val_data), MockDataset(test_data)


def generate_synthetic_audio(lang: str, duration: float, sample_rate: int, samples_per_audio: int) -> np.ndarray:
    """Generate synthetic audio signal for a given language."""
    # Create language-specific frequency patterns
    lang_seed = hash(lang) % 1000
    np.random.seed(lang_seed)
    
    # Generate pink noise with language-specific characteristics
    frequencies = np.random.rand(10) * 2000 + 500  # 500-2500 Hz range
    amplitudes = np.random.rand(10) * 0.5 + 0.1
    
    t = np.linspace(0, duration, samples_per_audio)
    signal = np.zeros(samples_per_audio)
    
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.05, samples_per_audio)
    signal += noise
    
    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    
    return signal


class AudioDataset(Dataset):
    """
    Custom dataset class for audio data.
    
    This dataset holds audio samples with their corresponding language labels.
    It's designed to work with PyTorch's DataLoader for efficient batch processing.
    """
    
    def __init__(self, data: List[Dict[str, Any]], audio_embedder=None):
        self.data = data
        self.audio_embedder = audio_embedder
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        if self.audio_embedder is not None:
            # Extract audio features on-the-fly
            audio_features = self.audio_embedder.extract_embeddings([item['audio']])[0]
            return {
                'audio_features': audio_features,
                'language': item['language'],
                'raw_language': item['raw_language']
            }
        else:
            return item


def load_and_split_data(
    seen_langs: List[str], 
    unseen_langs: List[str]
) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """
    Load and split the FLEURS dataset into train, validation, and test sets.
    
    This function first tries to load local audio files, then FLEURS dataset,
    and finally falls back to synthetic data.
    
    Args:
        seen_langs: List of language codes used for training
        unseen_langs: List of language codes used for testing
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    
    print("Loading FLEURS dataset...")
    
    # First, try to load local audio files
    print("🔍 Checking for local audio files...")
    local_datasets = load_local_audio_data()
    
    dataset_train = None
    dataset_validation = None
    dataset_test = None
    
    if local_datasets:
        print(f"✅ Found local audio data for {len(local_datasets)} languages")
        print(f"Available languages: {list(local_datasets.keys())}")
        
        # Use local data if available
        try:
            # Process local data similar to FLEURS format
            all_train_data = []
            all_val_data = []
            all_test_data = []
            
            for lang, items in local_datasets.items():
                for i, item in enumerate(items):
                    data_item = {
                        'audio': item['audio'],
                        'lang_id': lang,
                        'id': f"{lang}_{i}"
                    }
                    
                    # Assign to appropriate dataset based on seen/unseen status
                    if lang in seen_langs:
                        # Split seen languages 80/20 for train/val
                        if i < len(items) * 0.8:
                            all_train_data.append(data_item)
                        else:
                            all_val_data.append(data_item)
                    elif lang in unseen_langs:
                        # All unseen language data goes to test
                        all_test_data.append(data_item)
            
            # Create mock dataset objects
            class LocalMockDataset:
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __iter__(self):
                    return iter(self.data)
            
            dataset_train = LocalMockDataset(all_train_data)
            dataset_validation = LocalMockDataset(all_val_data)
            dataset_test = LocalMockDataset(all_test_data)
            
            print(f"✅ Successfully loaded local audio data:")
            print(f"  Training samples: {len(all_train_data)} (from seen languages)")
            print(f"  Validation samples: {len(all_val_data)} (from seen languages)")
            print(f"  Test samples: {len(all_test_data)} (from unseen languages)")
            
        except Exception as local_e:
            print(f"❌ Failed to process local audio data: {local_e}")
            dataset_train = None
    
    # If no local data worked, try loading FLEURS dataset
    if dataset_train is None:
        print("📡 No local audio found, attempting to load FLEURS dataset...")
        
        try:
            # Try different FLEURS loading approaches
            datasets_by_lang = {}
            sample_languages = seen_langs[:5] + unseen_langs[:3]
            
            print("Loading FLEURS dataset using modern HuggingFace API...")
            
            # Try simple direct loading first
            for lang in sample_languages:
                try:
                    lang_dataset = load_dataset("google/fleurs", lang, split="validation[:30]")
                    lang_data = list(lang_dataset)
                    if lang_data:
                        datasets_by_lang[lang] = lang_data
                        print(f"✅ Loaded {len(lang_data)} samples for {lang}")
                except Exception as lang_e:
                    print(f"❌ Failed to load {lang}: {lang_e}")
                    continue
            
            if datasets_by_lang:
                # Process FLEURS data into train/val/test
                all_train_data = []
                all_val_data = []
                all_test_data = []
                
                for lang, items in datasets_by_lang.items():
                    for i, item in enumerate(items):
                        data_item = {
                            'audio': item['audio'],
                            'lang_id': lang,
                            'id': f"{lang}_{i}"
                        }
                        
                        if lang in seen_langs:
                            # Split seen languages 80/20 for train/val
                            if i < len(items) * 0.8:
                                all_train_data.append(data_item)
                            else:
                                all_val_data.append(data_item)
                        elif lang in unseen_langs:
                            all_test_data.append(data_item)
                
                # Create dataset objects
                class FleursMockDataset:
                    def __init__(self, data):
                        self.data = data
                    def __len__(self):
                        return len(self.data)
                    def __iter__(self):
                        return iter(self.data)
                
                dataset_train = FleursMockDataset(all_train_data)
                dataset_validation = FleursMockDataset(all_val_data)
                dataset_test = FleursMockDataset(all_test_data)
                
                print(f"✅ Successfully loaded FLEURS data:")
                print(f"  Training samples: {len(all_train_data)} (from seen languages)")
                print(f"  Validation samples: {len(all_val_data)} (from seen languages)")
                print(f"  Test samples: {len(all_test_data)} (from unseen languages)")
            else:
                raise Exception("No FLEURS language datasets could be loaded")
                        
        except Exception as fleurs_error:
            print(f"❌ FLEURS loading failed: {fleurs_error}")
            dataset_train = None
    
    # Final fallback: create synthetic data
    if dataset_train is None:
        print("🎭 Creating synthetic dataset for demonstration...")
        
        dataset_train, dataset_validation, dataset_test = create_synthetic_dataset(seen_langs, unseen_langs)
        
        print("✅ Successfully created synthetic audio data")
    
    # Now process the dataset (whether local, FLEURS, or synthetic) into the final format
    train_data = []
    validation_data = []
    test_data = []
    
    # Process training data
    for item in dataset_train:
        train_data.append({
            'audio': item['audio'],
            'language': item['lang_id'],
            'raw_language': item['lang_id']
        })
    
    # Process validation data
    if dataset_validation is not None:
        for item in dataset_validation:
            validation_data.append({
                'audio': item['audio'],
                'language': item['lang_id'],
                'raw_language': item['lang_id']
            })
    
    # Process test data  
    if dataset_test is not None:
        for item in dataset_test:
            test_data.append({
                'audio': item['audio'],
                'language': item['lang_id'],
                'raw_language': item['lang_id']
            })
    
    # Apply filtering by seen/unseen languages and sample limits
    train_data = [item for item in train_data if item['language'] in seen_langs]
    validation_data = [item for item in validation_data if item['language'] in seen_langs]
    test_data = [item for item in test_data if item['language'] in unseen_langs]
    
    # Limit samples if specified
    if config.MAX_SAMPLES_PER_DATASET is not None and isinstance(config.MAX_SAMPLES_PER_DATASET, int):
        random.seed(config.RANDOM_SEED)
        
        max_samples = config.MAX_SAMPLES_PER_DATASET
        if len(train_data) > max_samples:
            train_data = random.sample(train_data, max_samples)
            
        if len(validation_data) > max_samples:
            validation_data = random.sample(validation_data, max_samples)
            
        if len(test_data) > max_samples:
            test_data = random.sample(test_data, max_samples)
    
    print(f"Final datasets:")
    print(f"  - Training: {len(train_data)} samples from seen languages")
    print(f"  - Validation: {len(validation_data)} samples from seen languages")
    print(f"  - Test: {len(test_data)} samples from unseen languages")
    
    # Count samples per language
    train_lang_counts = {}
    val_lang_counts = {}
    test_lang_counts = {}
    
    for item in train_data:
        lang = item['language']
        train_lang_counts[lang] = train_lang_counts.get(lang, 0) + 1
        
    for item in validation_data:
        lang = item['language']
        val_lang_counts[lang] = val_lang_counts.get(lang, 0) + 1
        
    for item in test_data:
        lang = item['language']
        test_lang_counts[lang] = test_lang_counts.get(lang, 0) + 1
    
    print(f"Training language distribution: {train_lang_counts}")
    print(f"Validation language distribution: {val_lang_counts}")
    print(f"Test language distribution: {test_lang_counts}")
    
    # Create dataset objects
    train_dataset = AudioDataset(train_data)
    validation_dataset = AudioDataset(validation_data)
    test_dataset = AudioDataset(test_data)
    
    return train_dataset, validation_dataset, test_dataset


def create_data_loaders(
    train_dataset: AudioDataset,
    validation_dataset: AudioDataset, 
    test_dataset: AudioDataset,
    batch_size: int | None = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for the datasets.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loading (uses config.BATCH_SIZE if None)
        
    Returns:
        Tuple of (train_loader, validation_loader, test_loader)
    """
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=False
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Created data loaders with batch size: {batch_size}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(validation_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, validation_loader, test_loader