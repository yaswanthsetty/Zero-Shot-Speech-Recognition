"""
Data preparation module for Zero-Shot Spoken Language Identification.

This module handles loading and splitting the FLEURS dataset for training
and evaluation purposes.
"""

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any
import random
from . import config


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
        Tuple of (train_dataset, validation_dataset)
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
    # Add some frequency characteristics to make languages "different"
    base_freq = hash(lang) % 1000 + 200  # Different base frequency per language
    t = np.linspace(0, duration, samples_per_audio)
    
    # Create synthetic "speech-like" audio with some randomness
    audio_signal = (
        0.3 * np.sin(2 * np.pi * base_freq * t) +
        0.2 * np.sin(2 * np.pi * (base_freq * 1.5) * t) +
        0.1 * np.random.randn(samples_per_audio)
    )
    
    # Apply envelope to make it more speech-like
    envelope = np.exp(-t / (duration * 0.3))
    audio_signal = audio_signal * envelope
    
    # Normalize
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    
    return audio_signal


class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for audio data with language labels.
    
    Args:
        data: List of dictionaries containing audio and language information
        audio_embedder: AudioEmbedder instance for feature extraction
    """
    
    def __init__(self, data: List[Dict[str, Any]], audio_embedder=None):
        self.data = data
        self.audio_embedder = audio_embedder
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # If audio embedder is provided, extract features
        if self.audio_embedder is not None:
            audio_features = self.audio_embedder.extract_embeddings(item['audio'])
            return {
                'audio_features': audio_features,
                'language': item['language'],
                'raw_language': item['raw_language']
            }
        else:
            # For datasets with pre-extracted features
            if 'audio_features' in item:
                return {
                    'audio_features': item['audio_features'],
                    'language': item['language'],
                    'raw_language': item['raw_language']
                }
            else:
                return {
                    'audio': item['audio'],
                    'language': item['language'],
                    'raw_language': item['raw_language']
                }


def load_and_split_data(
    seen_langs: List[str], 
    unseen_langs: List[str]
) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """
    Load and split the FLEURS dataset into train, validation, and test sets.
    
    This function loads the FLEURS dataset and creates three datasets:
    - Train dataset: Contains only audio from seen languages
    - Validation dataset: Contains only audio from seen languages  
    - Test dataset: Contains only audio from unseen languages (for zero-shot evaluation)
    
    Args:
        seen_langs: List of language codes used for training
        unseen_langs: List of language codes used for testing
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    
    print("Loading FLEURS dataset...")
    
    # Load the dataset
    try:
        # Try loading FLEURS without legacy script loading
        print("Attempting to load FLEURS dataset...")
        
        # Try different approaches for FLEURS
        try:
            # First attempt: load specific language subsets
            datasets_by_lang = {}
            sample_languages = seen_langs[:3] + unseen_langs[:2]  # Sample subset for demo
            
            for lang in sample_languages:
                try:
                    lang_dataset = load_dataset("google/fleurs", lang, split="train[:100]")
                    datasets_by_lang[lang] = lang_dataset
                    print(f"Loaded {len(lang_dataset)} samples for {lang}")
                except Exception as lang_e:
                    print(f"Failed to load {lang}: {lang_e}")
                    continue
            
            if datasets_by_lang:
                # Create combined dataset from language subsets
                all_train_data = []
                all_val_data = []
                
                for lang, dataset in datasets_by_lang.items():
                    for i, item in enumerate(dataset):
                        data_item = {
                            'audio': item['audio'],
                            'lang_id': lang,
                            'id': f"{lang}_{i}"
                        }
                        
                        # Split 80/20 for train/val
                        if i < len(dataset) * 0.8:
                            all_train_data.append(data_item)
                        else:
                            all_val_data.append(data_item)
                
                # Create mock dataset objects
                class MockDataset:
                    def __init__(self, data):
                        self.data = data
                    def __len__(self):
                        return len(self.data)
                    def __iter__(self):
                        return iter(self.data)
                
                dataset_train = MockDataset(all_train_data)
                dataset_validation = MockDataset(all_val_data)
                
                print(f"Successfully created combined dataset:")
                print(f"  Training samples: {len(all_train_data)}")
                print(f"  Validation samples: {len(all_val_data)}")
            else:
                raise Exception("No language datasets could be loaded")
                
        except Exception as fleurs_error:
            print(f"FLEURS loading failed: {fleurs_error}")
            raise Exception("Could not load FLEURS dataset")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset for demonstration...")
        
        # Create synthetic audio data for demonstration
        dataset_train, dataset_validation, dataset_test = create_synthetic_dataset(seen_langs, unseen_langs)
        
        # Process synthetic data
        train_data = []
        validation_data = []
        test_data = []
        
        # Process training data (already filtered for seen languages)
        for item in dataset_train:
            train_data.append({
                'audio': item['audio'],
                'language': item['lang_id'],
                'raw_language': item['lang_id']
            })
        
        # Process validation data (already filtered for seen languages)
        for item in dataset_validation:
            validation_data.append({
                'audio': item['audio'],
                'language': item['lang_id'],
                'raw_language': item['lang_id']
            })
        
        # Process test data (already filtered for unseen languages)
        for item in dataset_test:
            test_data.append({
                'audio': item['audio'],
                'language': item['lang_id'],
                'raw_language': item['lang_id']
            })
        
        print(f"Processed synthetic datasets:")
        print(f"  - Training: {len(train_data)} samples")
        print(f"  - Validation: {len(validation_data)} samples")
        print(f"  - Test: {len(test_data)} samples")
        
        # Skip the real dataset processing since we're using synthetic data
        synthetic_data_used = True
    
    if not 'synthetic_data_used' in locals():
        # Filter datasets by language (for real data)
        print("Filtering datasets by language...")
        
        # Filter training data for seen languages
        train_data = []
        validation_data = []
        test_data = []
        
        # Process training data
        for item in dataset_train:
            lang_code = item['lang_id']
            if lang_code in seen_langs:
                train_data.append({
                    'audio': item['audio'],
                    'language': lang_code,
                    'raw_language': item['lang_id']
                })
            elif lang_code in unseen_langs:
                test_data.append({
                    'audio': item['audio'],
                    'language': lang_code,
                    'raw_language': item['lang_id']
                })
        
        # Process validation data  
        for item in dataset_validation:
            lang_code = item['lang_id']
            if lang_code in seen_langs:
                validation_data.append({
                    'audio': item['audio'],
                    'language': lang_code,
                    'raw_language': item['lang_id']
                })
            elif lang_code in unseen_langs:
                test_data.append({
                    'audio': item['audio'],
                    'language': lang_code,
                    'raw_language': item['lang_id']
                })
    
    # Limit samples if specified
    if config.MAX_SAMPLES_PER_DATASET is not None:
        random.seed(config.RANDOM_SEED)
        
        if len(train_data) > config.MAX_SAMPLES_PER_DATASET:
            train_data = random.sample(train_data, config.MAX_SAMPLES_PER_DATASET)
            
        if len(validation_data) > config.MAX_SAMPLES_PER_DATASET:
            validation_data = random.sample(validation_data, config.MAX_SAMPLES_PER_DATASET)
            
        if len(test_data) > config.MAX_SAMPLES_PER_DATASET:
            test_data = random.sample(test_data, config.MAX_SAMPLES_PER_DATASET)
    
    print(f"Created datasets:")
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
    batch_size: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for the datasets.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for the data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Custom collate function to handle variable-length audio
    def collate_fn(batch):
        """Custom collate function for batching audio data."""
        if 'audio_features' in batch[0]:
            # If features are already extracted
            audio_features = torch.stack([item['audio_features'] for item in batch])
            languages = [item['language'] for item in batch]
            raw_languages = [item['raw_language'] for item in batch]
            
            return {
                'audio_features': audio_features,
                'languages': languages,
                'raw_languages': raw_languages
            }
        else:
            # If raw audio data
            audios = [item['audio'] for item in batch]
            languages = [item['language'] for item in batch]
            raw_languages = [item['raw_language'] for item in batch]
            
            return {
                'audios': audios,
                'languages': languages,
                'raw_languages': raw_languages
            }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Created data loaders with batch size: {batch_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader