#!/usr/bin/env python3
"""
Ultra-fast demonstration version - completes in 2-3 minutes.

Perfect for:
- Quick testing
- GitHub Codespaces
- First-time users
- Demonstration purposes

Usage:
    python run_fast_demo.py
"""

print("⚡ Ultra-Fast Zero-Shot Language Identification Demo")
print("🎯 Optimized for speed - completes in 2-3 minutes!")
print()

# Ultra-optimized configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Override configuration before importing anything else
import torch
from src import config

# Ultra-fast settings
config.FEATURE_EXTRACTION_BATCH_SIZE = 2
config.BATCH_SIZE = 4
config.MAX_SAMPLES_PER_DATASET = 10    # Only 10 samples per language
config.NUM_EPOCHS = 1                  # Single epoch
config.LEARNING_RATE = 1e-3            # Higher learning rate
config.HIDDEN_DIM = 128                # Smaller model
config.DEVICE = torch.device("cpu")    # Force CPU
config.LOG_INTERVAL = 1                # Log every batch

print("🔧 Ultra-fast configuration loaded:")
print(f"   - Samples per language: {config.MAX_SAMPLES_PER_DATASET}")
print(f"   - Training samples: ~50 total")  
print(f"   - Validation samples: ~10 total")
print(f"   - Test samples: ~15 total")
print(f"   - Epochs: {config.NUM_EPOCHS}")
print(f"   - Expected time: 2-3 minutes")
print("=" * 50)

# Import and run main
from main import main

if __name__ == "__main__":
    try:
        result = main()
        print("=" * 50)
        print("🎉 ULTRA-FAST DEMO COMPLETED!")
        print("✨ This demonstrates the complete zero-shot pipeline")
        print("📈 For better accuracy, use more samples and epochs")
        print("🔧 Edit config_optimized.py to customize settings")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try running 'python diagnose.py' to check your system")