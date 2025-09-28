"""
Configuration file for Zero-Shot Spoken Language Identification project.

This module contains all hyperparameters, data settings, model paths, and other 
configuration variables used throughout the project.

The configuration automatically adapts to the available system resources
for optimal performance in different environments (Codespaces, local, cloud).
"""

import torch
import os
import psutil
import gc

# ============================================================================
# Data Settings
# ============================================================================

# Dataset configuration
DATASET_NAME = "google/fleurs"

# Language configuration for zero-shot setup
# 15 languages for training (seen languages)
SEEN_LANGUAGES = [
    "en_us",    # English (US)
    "es_419",   # Spanish
    "fr_fr",    # French
    "de_de",    # German
    "it_it",    # Italian
    "pt_br",    # Portuguese (Brazil)
    "ru_ru",    # Russian
    "ja_jp",    # Japanese
    "ko_kr",    # Korean
    "zh_cn",    # Chinese (Simplified)
    "ar_eg",    # Arabic (Egypt)
    "hi_in",    # Hindi
    "tr_tr",    # Turkish
    "pl_pl",    # Polish
    "nl_nl"     # Dutch
]

# 5 languages for testing (unseen languages)
UNSEEN_LANGUAGES = [
    "sv_se",    # Swedish
    "da_dk",    # Danish
    "no_no",    # Norwegian
    "fi_fi",    # Finnish
    "cs_cz"     # Czech
]

# All languages combined
ALL_LANGUAGES = SEEN_LANGUAGES + UNSEEN_LANGUAGES

# Dataset split ratios
TRAIN_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.2

# Intelligent resource management based on available memory
def _get_memory_info():
    """Get system memory information for intelligent resource allocation."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    return available_gb, total_gb

def _calculate_optimal_settings():
    """Calculate optimal settings based on system resources."""
    available_gb, total_gb = _get_memory_info()
    
    # Conservative settings for different memory ranges
    if available_gb >= 12:  # High-end machine
        return {
            'max_samples': 1000,
            'batch_size': 32,
            'feature_batch_size': 16,
            'hidden_dim': 512,
            'num_epochs': 10
        }
    elif available_gb >= 8:  # Medium machine
        return {
            'max_samples': 500,
            'batch_size': 16,
            'feature_batch_size': 8,
            'hidden_dim': 256,
            'num_epochs': 5
        }
    elif available_gb >= 4:  # Codespaces/lightweight
        return {
            'max_samples': 200,
            'batch_size': 8,
            'feature_batch_size': 4,
            'hidden_dim': 128,
            'num_epochs': 3
        }
    else:  # Very constrained
        return {
            'max_samples': 100,
            'batch_size': 4,
            'feature_batch_size': 2,
            'hidden_dim': 64,
            'num_epochs': 2
        }

# Get optimal settings based on system resources
_optimal_settings = _calculate_optimal_settings()

# Sample limits - automatically adjusted based on available memory
MAX_SAMPLES_PER_DATASET = _optimal_settings['max_samples']

# ============================================================================
# Model Settings
# ============================================================================

# Pre-trained audio model
PRETRAINED_AUDIO_MODEL = "facebook/wav2vec2-base-960h"

# Model save path
MODEL_SAVE_PATH = "../models/projection_model.pth"

# Embedding dimensions
AUDIO_EMBEDDING_DIM = 768      # wav2vec2-base output dimension
PHONOLOGICAL_EMBEDDING_DIM = 22  # panphon feature dimension

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Training parameters - automatically adjusted for optimal performance
LEARNING_RATE = 1e-4
BATCH_SIZE = _optimal_settings['batch_size']
NUM_EPOCHS = _optimal_settings['num_epochs']

# Device configuration (automatically detect GPU availability)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer settings
WEIGHT_DECAY = 1e-5

# Model architecture parameters - automatically scaled
HIDDEN_DIM = _optimal_settings['hidden_dim']
DROPOUT_RATE = 0.3

# ============================================================================
# Audio Processing Settings
# ============================================================================

# Audio parameters
SAMPLE_RATE = 16000  # Standard sample rate for wav2vec2
MAX_AUDIO_LENGTH = 30.0  # Maximum audio length in seconds

# ============================================================================
# Evaluation Settings
# ============================================================================

# Top-k accuracy settings
TOP_K_VALUES = [1, 3]

# ============================================================================
# Paths and Directories
# ============================================================================

# Create models directory if it doesn't exist
MODELS_DIR = "../models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ============================================================================
# Logging and Monitoring
# ============================================================================

# Progress bar and logging settings
VERBOSE = True
LOG_INTERVAL = 5  # Log every N batches during training
FEATURE_EXTRACTION_BATCH_SIZE = _optimal_settings['feature_batch_size']  # Automatically scaled

# Random seed for reproducibility
RANDOM_SEED = 42

# Display configuration summary
available_gb, total_gb = _get_memory_info()
print(f"Configuration loaded. Using device: {DEVICE}")
print(f"System Memory: {available_gb:.1f}GB available / {total_gb:.1f}GB total")
print(f"Optimized Settings: batch_size={BATCH_SIZE}, hidden_dim={HIDDEN_DIM}, max_samples={MAX_SAMPLES_PER_DATASET}")
print(f"Seen languages: {len(SEEN_LANGUAGES)}")
print(f"Unseen languages: {len(UNSEEN_LANGUAGES)}")