"""
Feature extraction module for Zero-Shot Spoken Language Identification.

This module handles both audio feature extraction using pre-trained models
and phonological language embedding generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import Dict, List, Union, Any, cast
try:
    import panphon
    PANPHON_AVAILABLE = True
except ImportError:
    print("Warning: panphon not available. Using fallback phonological features.")
    PANPHON_AVAILABLE = False

import librosa
from . import config


class AudioEmbedder:
    """
    Audio feature extractor using pre-trained Wav2Vec2 model.
    
    This class loads a pre-trained Wav2Vec2 model and extracts fixed-size
    embeddings from variable-length audio inputs.
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize the AudioEmbedder.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        if model_name is None:
            model_name = config.PRETRAINED_AUDIO_MODEL
            
        print(f"Loading audio model: {model_name}")
        
        # Load the feature extractor and model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # Move model to appropriate device
        self.device = config.DEVICE
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()  # Set to evaluation mode
        
        print(f"Audio model loaded successfully on {self.device}")
    
    def extract_embeddings(self, audio_batch: Union[Dict, List]) -> torch.Tensor:
        """
        Extract audio embeddings from a batch of audio samples.
        
        Args:
            audio_batch: Batch of audio data (can be dict with 'array' key or list)
            
        Returns:
            torch.Tensor: Fixed-size audio embeddings of shape [batch_size, embedding_dim]
        """
        # Handle different input formats
        if isinstance(audio_batch, dict):
            if 'array' in audio_batch:
                audio_arrays = [audio_batch['array']]
                sample_rates = [audio_batch.get('sampling_rate', config.SAMPLE_RATE)]
            else:
                raise ValueError("Audio batch dict must contain 'array' key")
        elif isinstance(audio_batch, list):
            if len(audio_batch) > 0 and isinstance(audio_batch[0], dict):
                audio_arrays = [item['array'] for item in audio_batch]
                sample_rates = [item.get('sampling_rate', config.SAMPLE_RATE) for item in audio_batch]
            else:
                audio_arrays = audio_batch
                sample_rates = [config.SAMPLE_RATE] * len(audio_batch)
        else:
            # Single audio array
            audio_arrays = [audio_batch]
            sample_rates = [config.SAMPLE_RATE]
        
        # Resample audio to target sample rate if needed
        processed_audio = []
        for audio_array, sr in zip(audio_arrays, sample_rates):
            if sr != config.SAMPLE_RATE:
                # Resample using librosa
                audio_array = librosa.resample(
                    y=audio_array, 
                    orig_sr=sr, 
                    target_sr=config.SAMPLE_RATE
                )
            
            # Truncate or pad audio to max length
            max_length = int(config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE)
            if len(audio_array) > max_length:
                audio_array = audio_array[:max_length]
            elif len(audio_array) < max_length:
                # Pad with zeros
                audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
            
            processed_audio.append(audio_array)
        
        # Extract features using the feature extractor
        with torch.no_grad():
            # Process the audio through the feature extractor
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features from the model
            outputs = self.model(**inputs)
            
            # Get the last hidden states
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling across the time dimension
            embeddings = torch.mean(last_hidden_states, dim=1)
            
            # Move back to CPU if needed for further processing
            embeddings = embeddings.cpu()
            
        return embeddings.squeeze(0) if embeddings.shape[0] == 1 else embeddings
    
    def extract_embeddings_batch(self, audio_items: List[Dict], batch_size: int = 8) -> List[torch.Tensor]:
        """
        Extract embeddings from a list of audio items in batches for efficiency.
        
        Args:
            audio_items: List of dictionaries containing audio data
            batch_size: Number of samples to process at once
            
        Returns:
            List of embedding tensors
        """
        all_embeddings = []
        
        print(f"Processing {len(audio_items)} audio samples in batches of {batch_size}...")
        
        for i in range(0, len(audio_items), batch_size):
            batch = audio_items[i:i + batch_size]
            
            # Show progress
            if i % (batch_size * 10) == 0 or i + batch_size >= len(audio_items):
                print(f"  Processed {min(i + batch_size, len(audio_items))}/{len(audio_items)} samples")
            
            try:
                # Extract batch embeddings
                batch_embeddings = self.extract_embeddings(batch)
                
                # Handle single sample vs batch
                if batch_embeddings.dim() == 1:
                    all_embeddings.append(batch_embeddings)
                else:
                    for j in range(batch_embeddings.shape[0]):
                        all_embeddings.append(batch_embeddings[j])
                        
            except Exception as e:
                print(f"  Error processing batch {i//batch_size + 1}: {e}")
                # Add dummy embeddings for failed samples
                for _ in range(len(batch)):
                    dummy_embedding = torch.zeros(config.AUDIO_EMBEDDING_DIM)
                    all_embeddings.append(dummy_embedding)
        
        print(f"Feature extraction completed. Generated {len(all_embeddings)} embeddings.")
        return all_embeddings


def get_phonological_vectors(languages: List[str]) -> Dict[str, torch.Tensor]:
    """
    Generate phonological feature vectors for a list of languages.
    
    This function uses the panphon library to create unique phonological
    "footprints" for each language based on their phoneme inventories.
    
    Args:
        languages: List of language codes
        
    Returns:
        Dictionary mapping language codes to their phonological vectors
    """
    print(f"Generating phonological vectors for {len(languages)} languages...")
    
    # Initialize panphon if available
    if PANPHON_AVAILABLE:
        ft = panphon.FeatureTable()
    else:
        ft = None
    
    # Language to phoneme mappings (simplified for demonstration)
    # In a real implementation, you would use comprehensive phoneme inventories
    language_phonemes = {
        # Seen languages
        "en_us": ["p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "es_419": ["p", "b", "t", "d", "k", "g", "f", "β", "s", "x", "m", "n", "ɲ", "l", "r", "rr", "w", "j"],
        "fr_fr": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "r", "w", "ɥ", "j"],
        "de_de": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "h", "m", "n", "ŋ", "l", "r", "j"],
        "it_it": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "m", "n", "ɲ", "l", "r", "w", "j"],
        "pt_br": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "r", "w", "j"],
        "ru_ru": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "m", "n", "l", "r", "j"],
        "ja_jp": ["p", "b", "t", "d", "k", "g", "s", "z", "ʃ", "h", "m", "n", "r", "w", "j"],
        "ko_kr": ["p", "b", "t", "d", "k", "g", "s", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "zh_cn": ["p", "b", "t", "d", "k", "g", "f", "s", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "ar_eg": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "ħ", "m", "n", "l", "r", "w", "j"],
        "hi_in": ["p", "b", "t", "d", "k", "g", "f", "s", "ʃ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "tr_tr": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "r", "j"],
        "pl_pl": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "m", "n", "ɲ", "l", "r", "w", "j"],
        "nl_nl": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        
        # Unseen languages
        "sv_se": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "ʃ", "h", "m", "n", "ŋ", "l", "r", "j"],
        "da_dk": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "no_no": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "ʃ", "h", "m", "n", "ŋ", "l", "r", "j"],
        "fi_fi": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "cs_cz": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "h", "m", "n", "ɲ", "l", "r", "j"]
    }
    
    phonological_vectors = {}
    
    for lang in languages:
        if lang in language_phonemes:
            phonemes = language_phonemes[lang]
        else:
            # Default phoneme set for unknown languages
            phonemes = ["p", "t", "k", "s", "m", "n", "l", "r"]
            print(f"Warning: Unknown language {lang}, using default phonemes")
        
        # Get phonological features for each phoneme
        feature_vectors = []
        if PANPHON_AVAILABLE and ft is not None:
            for phoneme in phonemes:
                try:
                    # Get feature vector for the phoneme
                    features = ft.fts(phoneme)
                    if features:
                        feature_vectors.append(features)
                except:
                    # Skip phonemes that can't be processed
                    continue
        else:
            # Fallback: create synthetic features based on phoneme count and language hash
            num_features = len(phonemes)
            feature_vectors = [[1.0 if i % 3 == 0 else 0.0 for i in range(config.PHONOLOGICAL_EMBEDDING_DIM)] for _ in range(num_features)]
        
        if feature_vectors:
            try:
                # Convert feature vectors to numeric arrays
                numeric_features = []
                for fv in feature_vectors:
                    if isinstance(fv, (list, tuple)):
                        # Convert to numeric values (assuming binary or numeric features)
                        numeric_fv = []
                        for val in fv:
                            if isinstance(val, str):
                                # Convert string features to binary (present/absent)
                                numeric_fv.append(1.0 if val in ['+', '1', 'true', 'True'] else 0.0)
                            elif isinstance(val, (int, float)):
                                numeric_fv.append(float(val))
                            else:
                                numeric_fv.append(0.0)
                        numeric_features.append(numeric_fv)
                    else:
                        # Skip invalid feature vectors
                        continue
                
                if numeric_features:
                    # Convert to numpy array and take element-wise maximum
                    features_array = np.array(numeric_features, dtype=np.float32)
                    
                    # Create phonological footprint by taking max across all phonemes
                    phonological_footprint = np.max(features_array, axis=0)
                    
                    # Ensure we have exactly the expected dimension
                    if len(phonological_footprint) > config.PHONOLOGICAL_EMBEDDING_DIM:
                        phonological_footprint = phonological_footprint[:config.PHONOLOGICAL_EMBEDDING_DIM]
                    elif len(phonological_footprint) < config.PHONOLOGICAL_EMBEDDING_DIM:
                        # Pad with zeros
                        padding = config.PHONOLOGICAL_EMBEDDING_DIM - len(phonological_footprint)
                        phonological_footprint = np.concatenate([
                            phonological_footprint, 
                            np.zeros(padding, dtype=np.float32)
                        ])
                    
                    # Convert to PyTorch tensor
                    phonological_vectors[lang] = torch.tensor(phonological_footprint, dtype=torch.float32)
                else:
                    raise ValueError("No valid numeric features found")
                    
            except Exception as feature_error:
                print(f"Warning: Feature processing failed for {lang}, using random vector: {feature_error}")
                # Fallback: create a random but consistent vector
                np.random.seed(hash(lang) % (2**32))
                random_vector = np.random.randn(config.PHONOLOGICAL_EMBEDDING_DIM).astype(np.float32)
                phonological_vectors[lang] = torch.tensor(random_vector, dtype=torch.float32)
        else:
            # Fallback: create a random but consistent vector
            print(f"Warning: Could not generate features for {lang}, using random vector")
            np.random.seed(hash(lang) % (2**32))  # Consistent random seed based on language
            random_vector = np.random.randn(config.PHONOLOGICAL_EMBEDDING_DIM).astype(np.float32)
            phonological_vectors[lang] = torch.tensor(random_vector, dtype=torch.float32)
    
    print(f"Generated phonological vectors for {len(phonological_vectors)} languages")
    
    # Print some statistics
    for lang, vector in list(phonological_vectors.items())[:3]:
        print(f"  {lang}: shape={vector.shape}, mean={vector.mean():.3f}, std={vector.std():.3f}")
    
    return phonological_vectors


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalize embeddings to unit length for cosine similarity computation.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        Normalized embeddings
    """
    return F.normalize(embeddings, p=2, dim=-1)


def compute_cosine_similarity(
    embeddings1: torch.Tensor, 
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [M, D]
        
    Returns:
        Cosine similarity matrix [N, M]
    """
    # Normalize embeddings
    embeddings1_norm = normalize_embeddings(embeddings1)
    embeddings2_norm = normalize_embeddings(embeddings2)
    
    # Compute cosine similarity
    similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())
    
    return similarity