#!/usr/bin/env python3
"""
Optimized configuration for resource-constrained environments.

This configuration is specifically designed for:
- GitHub Codespaces
- Limited memory systems
- CPU-only execution
- Quick demonstration runs

Usage:
    python -c "import config_optimized; exec(open('main.py').read())"
"""

# Import base configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import *

# Override with optimized settings
print("🔧 Loading optimized configuration for limited resources...")

# ============================================================================
# OPTIMIZED SETTINGS FOR RESOURCE-CONSTRAINED ENVIRONMENTS
# ============================================================================

# Reduce memory usage dramatically
FEATURE_EXTRACTION_BATCH_SIZE = 2    # Instead of 8 (75% reduction)
BATCH_SIZE = 8                       # Instead of 32 (75% reduction)
MAX_SAMPLES_PER_DATASET = 20         # Instead of 500 (96% reduction)

# Faster completion
NUM_EPOCHS = 2                       # Instead of 10 (80% reduction)
LEARNING_RATE = 5e-4                 # Higher LR for faster convergence

# Lighter model
HIDDEN_DIM = 256                     # Instead of 512 (50% reduction)

# More frequent progress updates
LOG_INTERVAL = 1                     # Log every batch

# Ensure CPU usage
DEVICE = torch.device("cpu")         # Force CPU to avoid GPU memory issues

print(f"✅ Optimized configuration loaded:")
print(f"   - Batch size: {BATCH_SIZE} (was 32)")
print(f"   - Feature batch size: {FEATURE_EXTRACTION_BATCH_SIZE} (was 8)")
print(f"   - Max samples: {MAX_SAMPLES_PER_DATASET} (was 500)")
print(f"   - Epochs: {NUM_EPOCHS} (was 10)")
print(f"   - Device: {DEVICE}")
print(f"   - Expected runtime: 5-10 minutes")
print(f"   - Memory usage: ~2GB (was ~8GB)")
print()