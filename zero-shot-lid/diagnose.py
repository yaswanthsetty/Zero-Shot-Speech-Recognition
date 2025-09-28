#!/usr/bin/env python3
"""
System diagnostic script for troubleshooting Zero-Shot Language Identification issues.

Checks:
- System resources (memory, disk, CPU)
- Python environment and dependencies
- PyTorch configuration
- Model loading capabilities
- Synthetic data generation

Usage:
    python diagnose.py
"""

import torch
import psutil
import sys
import os
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_status(check, status, details=""):
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {check:<40} {details}")

def diagnose_system():
    print_header("SYSTEM DIAGNOSTICS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System resources
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"\n📊 SYSTEM RESOURCES:")
    print(f"   Memory: {memory.used/1024**3:.1f}GB used / {memory.total/1024**3:.1f}GB total ({memory.percent:.1f}%)")
    print(f"   Disk:   {disk.used/1024**3:.1f}GB used / {disk.total/1024**3:.1f}GB total ({disk.percent:.1f}%)")
    print(f"   CPU:    {psutil.cpu_count()} cores, {psutil.cpu_percent(interval=1):.1f}% usage")
    
    # Memory recommendations
    if memory.available < 4 * 1024**3:  # Less than 4GB available
        print("⚠️  LOW MEMORY: Recommend using optimized configuration")
    else:
        print("✅ MEMORY: Sufficient for standard configuration")

def diagnose_python():
    print_header("PYTHON ENVIRONMENT")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check key dependencies
    dependencies = {
        'torch': 'PyTorch deep learning framework',
        'transformers': 'HuggingFace transformers library',
        'datasets': 'HuggingFace datasets library',
        'librosa': 'Audio processing library',
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing',
        'tqdm': 'Progress bars'
    }
    
    print(f"\n📦 DEPENDENCIES:")
    for dep, description in dependencies.items():
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{dep} ({description})", True, f"v{version}")
        except ImportError:
            print_status(f"{dep} ({description})", False, "NOT INSTALLED")
    
    # Check optional dependencies
    optional_deps = {
        'panphon': 'Phonological features (has fallback)'
    }
    
    print(f"\n📦 OPTIONAL DEPENDENCIES:")
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print_status(f"{dep} ({description})", True, "available")
        except ImportError:
            print_status(f"{dep} ({description})", False, "using fallback")

def diagnose_pytorch():
    print_header("PYTORCH CONFIGURATION")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
        device = torch.device("cuda")
    else:
        print("Using CPU device (recommended for Codespaces)")
        device = torch.device("cpu")
    
    print(f"Recommended device: {device}")
    
    # Test tensor operations
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.mm(x, x.t())
        print_status("Basic tensor operations", True, f"shape {y.shape}")
    except Exception as e:
        print_status("Basic tensor operations", False, str(e))

def diagnose_model_loading():
    print_header("MODEL LOADING TEST")
    
    try:
        print("Testing Wav2Vec2 model loading...")
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        
        # Test feature extractor
        print("  Loading feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        print_status("Feature extractor", True, "loaded successfully")
        
        # Test model loading
        print("  Loading Wav2Vec2 model...")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        print_status("Wav2Vec2 model", True, f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
        
        # Test model inference
        print("  Testing model inference...")
        import numpy as np
        dummy_audio = np.random.randn(16000)  # 1 second of audio
        inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print_status("Model inference", True, f"output shape {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print_status("Model loading/inference", False, str(e))
        print("💡 This might be the cause of termination. Try optimized configuration.")

def diagnose_synthetic_data():
    print_header("SYNTHETIC DATA GENERATION")
    
    try:
        # Test synthetic audio generation
        print("Testing synthetic audio generation...")
        import numpy as np
        
        duration = 3.0
        sample_rate = 16000
        samples = int(sample_rate * duration)
        
        # Generate synthetic audio (similar to the actual implementation)
        t = np.linspace(0, duration, samples)
        audio_signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 300 * t) +
            0.1 * np.random.randn(samples)
        )
        
        print_status("Synthetic audio generation", True, f"{len(audio_signal)} samples, {duration}s")
        
        # Test multiple languages
        languages = ['en_us', 'es_419', 'fr_fr']
        for lang in languages:
            # Language-specific frequency
            base_freq = hash(lang) % 1000 + 200
            lang_audio = 0.3 * np.sin(2 * np.pi * base_freq * t)
            print_status(f"Language {lang}", True, f"base_freq={base_freq}Hz")
            
    except Exception as e:
        print_status("Synthetic data generation", False, str(e))

def main():
    print("🔍 Zero-Shot Language Identification - System Diagnostics")
    print("This script will help identify issues with your setup")
    
    diagnose_system()
    diagnose_python()
    diagnose_pytorch()
    diagnose_model_loading()
    diagnose_synthetic_data()
    
    print_header("RECOMMENDATIONS")
    
    memory = psutil.virtual_memory()
    if memory.available < 4 * 1024**3:
        print("⚠️  MEMORY: Use optimized configuration")
        print("   python run_optimized.py")
    else:
        print("✅ MEMORY: Can use standard configuration")
        print("   python main.py")
    
    if not torch.cuda.is_available():
        print("💡 GPU: Using CPU (normal for Codespaces)")
    
    print("\n🎯 NEXT STEPS:")
    print("1. If model loading failed → Use optimized configuration")
    print("2. If memory is low → Reduce batch sizes further")
    print("3. If everything passed → Try running the main program")
    print("4. If still having issues → Check the troubleshooting section in README")

if __name__ == "__main__":
    main()