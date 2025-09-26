"""
Configuration file for Zero-Shot Spoken Language Identification project.

This module contains all hyperparameters, data settings, model paths, and other 
configuration variables used throughout the project.
"""

import torch
import os

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

# Sample limits for demonstration (set to None for full dataset)
MAX_SAMPLES_PER_DATASET = 500

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

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Device configuration (automatically detect GPU availability)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizer settings
WEIGHT_DECAY = 1e-5

# Model architecture parameters
HIDDEN_DIM = 512
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
LOG_INTERVAL = 10  # Log every N batches during training

# Random seed for reproducibility
RANDOM_SEED = 42

print(f"Configuration loaded. Using device: {DEVICE}")
print(f"Seen languages: {len(SEEN_LANGUAGES)}")
print(f"Unseen languages: {len(UNSEEN_LANGUAGES)}")