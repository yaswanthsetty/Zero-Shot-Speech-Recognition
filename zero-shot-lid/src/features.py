"""
Feature extraction module for Zero-Shot Spoken Language Identification.

This module handles both audio feature extraction using pre-trained models
and phonological language embedding generation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
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
        Includes smart memory management and error recovery.
        
        Args:
            audio_items: List of dictionaries containing audio data
            batch_size: Number of samples to process at once
            
        Returns:
            List of embedding tensors
        """
        all_embeddings = []
        original_batch_size = batch_size
        
        print(f"Processing {len(audio_items)} audio samples in batches of {batch_size}...")
        
        for i in range(0, len(audio_items), batch_size):
            batch = audio_items[i:i + batch_size]
            
            # Show progress
            if i % (batch_size * 5) == 0 or i + batch_size >= len(audio_items):
                print(f"  Processed {min(i + batch_size, len(audio_items))}/{len(audio_items)} samples")
            
            try:
                # Extract batch embeddings
                with torch.no_grad():  # Disable gradient computation for memory efficiency
                    batch_embeddings = self.extract_embeddings(batch)
                
                # Handle single sample vs batch
                if batch_embeddings.dim() == 1:
                    all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory
                else:
                    for j in range(batch_embeddings.shape[0]):
                        all_embeddings.append(batch_embeddings[j].cpu())
                
                # Clean up memory after each batch
                del batch_embeddings
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Memory error in batch {i//batch_size + 1}, reducing batch size...")
                    # Try with smaller batch size
                    batch_size = max(1, batch_size // 2)
                    
                    # Process failed batch with smaller size
                    for j in range(0, len(batch), batch_size):
                        mini_batch = batch[j:j + batch_size]
                        try:
                            with torch.no_grad():
                                mini_embeddings = self.extract_embeddings(mini_batch)
                            if mini_embeddings.dim() == 1:
                                all_embeddings.append(mini_embeddings.cpu())
                            else:
                                for k in range(mini_embeddings.shape[0]):
                                    all_embeddings.append(mini_embeddings[k].cpu())
                            del mini_embeddings
                            gc.collect()
                        except Exception as inner_e:
                            print(f"    Failed to process mini-batch: {inner_e}")
                            # Add dummy embeddings for failed samples
                            for _ in range(len(mini_batch)):
                                dummy_embedding = torch.zeros(config.AUDIO_EMBEDDING_DIM)
                                all_embeddings.append(dummy_embedding)
                else:
                    print(f"  Error processing batch {i//batch_size + 1}: {e}")
                    # Add dummy embeddings for failed samples
                    for _ in range(len(batch)):
                        dummy_embedding = torch.zeros(config.AUDIO_EMBEDDING_DIM)
                        all_embeddings.append(dummy_embedding)
        
        print(f"Feature extraction completed. Generated {len(all_embeddings)} embeddings.")
        if batch_size != original_batch_size:
            print(f"Note: Batch size was automatically reduced to {batch_size} due to memory constraints.")
        
        return all_embeddings


def get_phonological_vectors(languages: List[str]) -> Dict[str, torch.Tensor]:
    """
    Generate phonological feature vectors for a list of languages.
    
    This function creates unique phonological "footprints" for each language 
    based on their phoneme inventories using a robust approach that works
    with or without the panphon library.
    
    Args:
        languages: List of language codes
        
    Returns:
        Dictionary mapping language codes to their phonological vectors
    """
    print(f"Generating phonological vectors for {len(languages)} languages...")
    
    # Initialize panphon if available and working
    ft = None
    panphon_working = False
    if PANPHON_AVAILABLE:
        try:
            ft = panphon.FeatureTable()
            # Test if it's working
            test_features = ft.word_to_vector_list("test")
            if test_features and len(test_features) > 0:
                panphon_working = True
                print("✅ Using panphon library for phonological features")
            else:
                print("⚠️  panphon library available but not working properly")
        except Exception as e:
            print(f"⚠️  panphon library failed to initialize: {e}")
    else:
        print("⚠️  panphon library not available")
    
    # Comprehensive phonological feature system
    # Each phoneme is mapped to a binary feature vector representing phonological properties
    phoneme_features = {
        # Consonants - [consonantal, sonorant, voice, continuant, nasal, labial, coronal, dorsal, anterior, strident]
        "p": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # voiceless bilabial stop
        "b": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # voiced bilabial stop
        "t": [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # voiceless alveolar stop
        "d": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],  # voiced alveolar stop
        "k": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # voiceless velar stop
        "g": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # voiced velar stop
        "f": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],  # voiceless labiodental fricative
        "v": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],  # voiced labiodental fricative
        "θ": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # voiceless dental fricative (English th)
        "ð": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],  # voiced dental fricative (English th)
        "s": [1, 0, 0, 1, 0, 0, 1, 0, 1, 1],  # voiceless alveolar fricative
        "z": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],  # voiced alveolar fricative
        "ʃ": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # voiceless postalveolar fricative
        "ʒ": [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # voiced postalveolar fricative
        "x": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # voiceless velar fricative
        "h": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless glottal fricative
        "m": [1, 1, 1, 0, 1, 1, 0, 0, 1, 0],  # bilabial nasal
        "n": [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],  # alveolar nasal
        "ŋ": [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],  # velar nasal
        "ɲ": [1, 1, 1, 0, 1, 0, 1, 0, 0, 0],  # palatal nasal
        "l": [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # alveolar lateral
        "r": [1, 1, 1, 1, 0, 0, 1, 0, 1, 0],  # alveolar trill/rhotic
        "w": [0, 1, 1, 1, 0, 1, 0, 1, 0, 0],  # labio-velar glide
        "j": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],  # palatal glide
        "β": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],  # voiced bilabial fricative
        "ɥ": [0, 1, 1, 1, 0, 1, 1, 0, 0, 0],  # labio-palatal glide
        "rr": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0], # alveolar trill (Spanish)
        "ħ": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # voiceless pharyngeal fricative
        # Add more features for language identification
        "tonal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # tone feature (12 dimensions total)
        "click": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # click feature
    }
    
    # Language to phoneme mappings with more accurate inventories
    language_phonemes = {
        # Indo-European languages (seen)
        "en_us": ["p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "es_419": ["p", "b", "t", "d", "k", "g", "f", "β", "s", "x", "m", "n", "ɲ", "l", "r", "rr", "w", "j"],
        "fr_fr": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "r", "w", "ɥ", "j"],
        "de_de": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "h", "m", "n", "ŋ", "l", "r", "j"],
        "it_it": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "m", "n", "ɲ", "l", "r", "w", "j"],
        "pt_br": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "r", "w", "j"],
        "ru_ru": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "m", "n", "l", "r", "j"],
        "pl_pl": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "m", "n", "ŋ", "l", "r", "w", "j"],
        "nl_nl": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "hi_in": ["p", "b", "t", "d", "k", "g", "f", "s", "ʃ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        
        # East Asian languages (seen)
        "ja_jp": ["p", "b", "t", "d", "k", "g", "s", "z", "ʃ", "h", "m", "n", "r", "w", "j"],
        "ko_kr": ["p", "b", "t", "d", "k", "g", "s", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "zh_cn": ["p", "b", "t", "d", "k", "g", "f", "s", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j", "tonal"],
        
        # Afro-Asiatic (seen)
        "ar_eg": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "ħ", "m", "n", "l", "r", "w", "j"],
        
        # Turkic (seen)
        "tr_tr": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        
        # Germanic languages (unseen - should be similar to seen Germanic languages)
        "sv_se": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "da_dk": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        "no_no": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        
        # Finno-Ugric (unseen - different family)
        "fi_fi": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
        
        # Slavic (unseen - similar to seen Slavic)
        "cs_cz": ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "x", "h", "m", "n", "ŋ", "l", "r", "w", "j"],
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
        
        # Try to generate real phonological features
        try:
            if panphon_working and ft is not None:
                # Try panphon first
                print(f"  Using panphon for {lang}...")
                feature_matrix = []
                for phoneme in phonemes:
                    try:
                        features = ft.fts(phoneme)
                        if features and len(features) > 0:
                            # panphon features are already in numeric format (list of ints)
                            # Convert directly to float list
                            if isinstance(features, (list, tuple)):
                                feature_matrix.append([float(f) for f in features])
                            else:
                                # Handle other formats that might be returned
                                feature_list = list(features) if hasattr(features, '__iter__') else []
                                if feature_list:
                                    feature_matrix.append([float(f) for f in feature_list])
                    except Exception:
                        continue
                
                if feature_matrix and len(feature_matrix) > 0:
                    features_array = np.array(feature_matrix, dtype=np.float32)
                    # Create language footprint using mean (percentage of phonemes with each feature)
                    phonological_footprint = np.mean(features_array, axis=0)
                    
                    # Check if we actually got meaningful features
                    if np.sum(np.abs(phonological_footprint)) > 0:
                        # Ensure correct dimensionality
                        if len(phonological_footprint) > config.PHONOLOGICAL_EMBEDDING_DIM:
                            phonological_footprint = phonological_footprint[:config.PHONOLOGICAL_EMBEDDING_DIM]
                        elif len(phonological_footprint) < config.PHONOLOGICAL_EMBEDDING_DIM:
                            padding = config.PHONOLOGICAL_EMBEDDING_DIM - len(phonological_footprint)
                            phonological_footprint = np.concatenate([
                                phonological_footprint, 
                                np.zeros(padding, dtype=np.float32)
                            ])
                        
                        phonological_vectors[lang] = torch.tensor(phonological_footprint, dtype=torch.float32)
                        print(f"✅ Real panphon features for {lang} (sum: {np.sum(np.abs(phonological_footprint)):.3f})")
                        continue
                    else:
                        print(f"⚠️  Panphon features for {lang} are all zeros, falling back to built-in")
            
            # Use built-in phonological feature system
            print(f"  Using built-in features for {lang}...")
            feature_matrix = []
            
            for phoneme in phonemes:
                if phoneme in phoneme_features:
                    feature_matrix.append(phoneme_features[phoneme])
                else:
                    # Unknown phoneme - use a neutral feature vector
                    neutral_features = [0] * 12  # All features absent
                    feature_matrix.append(neutral_features)
            
            if feature_matrix:
                features_array = np.array(feature_matrix, dtype=np.float32)
                
                # Create rich language representation
                feature_stats = np.concatenate([
                    np.mean(features_array, axis=0),      # Feature presence rates
                    np.std(features_array, axis=0),       # Feature variability
                ])
                
                # Normalize to expected dimensionality
                if len(feature_stats) > config.PHONOLOGICAL_EMBEDDING_DIM:
                    phonological_footprint = feature_stats[:config.PHONOLOGICAL_EMBEDDING_DIM]
                elif len(feature_stats) < config.PHONOLOGICAL_EMBEDDING_DIM:
                    padding = config.PHONOLOGICAL_EMBEDDING_DIM - len(feature_stats)
                    phonological_footprint = np.concatenate([
                        feature_stats,
                        np.zeros(padding, dtype=np.float32)
                    ])
                else:
                    phonological_footprint = feature_stats
                
                phonological_vectors[lang] = torch.tensor(phonological_footprint, dtype=torch.float32)
                print(f"✅ Built-in phonological features for {lang}")
            else:
                raise ValueError("No phonemes could be processed")
                
        except Exception as e:
            print(f"❌ Feature generation failed for {lang}: {e}")
            # Intelligent fallback with language family structure
            np.random.seed(hash(lang) % (2**32))
            
            # Language family biases for better zero-shot performance
            family_bias = np.zeros(config.PHONOLOGICAL_EMBEDDING_DIM, dtype=np.float32)
            
            if lang in ["en_us", "de_de", "nl_nl", "sv_se", "da_dk", "no_no"]:  # Germanic
                family_bias[:5] = [0.8, 0.6, 0.4, 0.7, 0.5]  # Germanic feature pattern
            elif lang in ["es_419", "fr_fr", "it_it", "pt_br"]:  # Romance  
                family_bias[:5] = [0.9, 0.3, 0.8, 0.6, 0.7]  # Romance feature pattern
            elif lang in ["ru_ru", "pl_pl", "cs_cz"]:  # Slavic
                family_bias[:5] = [0.7, 0.8, 0.5, 0.9, 0.4]  # Slavic feature pattern
            elif lang in ["ja_jp", "ko_kr", "zh_cn"]:  # East Asian
                family_bias[:5] = [0.5, 0.4, 0.9, 0.3, 0.8]  # East Asian feature pattern
            elif lang in ["fi_fi"]:  # Finno-Ugric
                family_bias[:5] = [0.6, 0.7, 0.3, 0.8, 0.9]  # Finno-Ugric feature pattern
            
            # Add some random variation
            random_component = np.random.randn(config.PHONOLOGICAL_EMBEDDING_DIM).astype(np.float32) * 0.1
            phonological_footprint = family_bias + random_component
            
            phonological_vectors[lang] = torch.tensor(phonological_footprint, dtype=torch.float32)
            print(f"⚠️  Using family-structured vector for {lang}")
    
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