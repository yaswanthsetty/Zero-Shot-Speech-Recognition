#!/usr/bin/env python3
"""
Main orchestration script for Zero-Shot Spoken Language Identification.

This script ties together all components of the project:
1. Data loading and preprocessing
2. Feature extraction (audio and phonological)
3. Model training
4. Zero-shot evaluation

Usage:
    python main.py
"""

import torch
import random
import numpy as np
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from src import config
from src.data_prep import load_and_split_data, create_data_loaders, AudioDataset
from src.features import AudioEmbedder, get_phonological_vectors
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_zero_shot, generate_evaluation_report


def set_random_seeds(seed: int = None):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = config.RANDOM_SEED
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to: {seed}")


def main():
    """Main execution function."""
    print("="*80)
    print("ZERO-SHOT SPOKEN LANGUAGE IDENTIFICATION")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Print configuration information
    print("CONFIGURATION:")
    print("-" * 40)
    print(f"Device: {config.DEVICE}")
    print(f"Seen languages: {len(config.SEEN_LANGUAGES)}")
    print(f"Unseen languages: {len(config.UNSEEN_LANGUAGES)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print(f"Max samples per dataset: {config.MAX_SAMPLES_PER_DATASET}")
    print()
    
    try:
        # Step 1: Load and split data
        print("STEP 1: LOADING AND SPLITTING DATA")
        print("-" * 40)
        
        train_dataset, val_dataset, test_dataset = load_and_split_data(
            seen_langs=config.SEEN_LANGUAGES,
            unseen_langs=config.UNSEEN_LANGUAGES
        )
        print()
        
        # Step 2: Initialize audio embedder and extract features
        print("STEP 2: EXTRACTING AUDIO FEATURES")
        print("-" * 40)
        
        audio_embedder = AudioEmbedder()
        
        # Extract features for all datasets
        print("Extracting features for training data...")
        train_data_with_features = []
        for i, item in enumerate(train_dataset.data):
            if i % 100 == 0:
                print(f"Processing training sample {i+1}/{len(train_dataset.data)}")
            
            try:
                audio_features = audio_embedder.extract_embeddings(item['audio'])
                train_data_with_features.append({
                    'audio_features': audio_features,
                    'language': item['language'],
                    'raw_language': item['raw_language']
                })
            except Exception as e:
                print(f"Error processing training sample {i}: {e}")
                continue
        
        print("Extracting features for validation data...")
        val_data_with_features = []
        for i, item in enumerate(val_dataset.data):
            if i % 50 == 0:
                print(f"Processing validation sample {i+1}/{len(val_dataset.data)}")
            
            try:
                audio_features = audio_embedder.extract_embeddings(item['audio'])
                val_data_with_features.append({
                    'audio_features': audio_features,
                    'language': item['language'],
                    'raw_language': item['raw_language']
                })
            except Exception as e:
                print(f"Error processing validation sample {i}: {e}")
                continue
        
        print("Extracting features for test data...")
        test_data_with_features = []
        for i, item in enumerate(test_dataset.data):
            if i % 50 == 0:
                print(f"Processing test sample {i+1}/{len(test_dataset.data)}")
            
            try:
                audio_features = audio_embedder.extract_embeddings(item['audio'])
                test_data_with_features.append({
                    'audio_features': audio_features,
                    'language': item['language'],
                    'raw_language': item['raw_language']
                })
            except Exception as e:
                print(f"Error processing test sample {i}: {e}")
                continue
        
        # Create new datasets with features
        # Add the audio features to the original datasets
        for i, features in enumerate(train_data_with_features):
            train_dataset.data[i]['audio_features'] = features['audio_features']
        for i, features in enumerate(val_data_with_features):
            val_dataset.data[i]['audio_features'] = features['audio_features']
        for i, features in enumerate(test_data_with_features):
            test_dataset.data[i]['audio_features'] = features['audio_features']
        
        train_dataset_features = train_dataset
        val_dataset_features = val_dataset
        test_dataset_features = test_dataset
        
        print(f"Feature extraction completed!")
        print(f"  Training samples with features: {len(train_data_with_features)}")
        print(f"  Validation samples with features: {len(val_data_with_features)}")
        print(f"  Test samples with features: {len(test_data_with_features)}")
        print()
        
        # Step 3: Generate phonological vectors
        print("STEP 3: GENERATING PHONOLOGICAL VECTORS")
        print("-" * 40)
        
        all_language_vectors = get_phonological_vectors(config.ALL_LANGUAGES)
        print()
        
        # Step 4: Create data loaders
        print("STEP 4: CREATING DATA LOADERS")
        print("-" * 40)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset_features, 
            val_dataset_features, 
            test_dataset_features
        )
        print()
        
        # Step 5: Create and initialize model
        print("STEP 5: CREATING MODEL")
        print("-" * 40)
        
        model = create_model()
        print()
        
        # Step 6: Train the model
        print("STEP 6: TRAINING MODEL")
        print("-" * 40)
        
        # Only use phonological vectors for seen languages during training
        seen_language_vectors = {
            lang: all_language_vectors[lang] 
            for lang in config.SEEN_LANGUAGES 
            if lang in all_language_vectors
        }
        
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            phonological_vectors=seen_language_vectors
        )
        print()
        
        # Step 7: Evaluate on zero-shot languages
        print("STEP 7: ZERO-SHOT EVALUATION")
        print("-" * 40)
        
        evaluation_results = evaluate_zero_shot(
            model=trained_model,
            test_loader=test_loader,
            all_language_vectors=all_language_vectors,
            verbose=True
        )
        print()
        
        # Step 8: Generate and display final results
        print("STEP 8: FINAL RESULTS")
        print("-" * 40)
        
        # Generate evaluation report
        report = generate_evaluation_report(
            evaluation_results, 
            save_path="../models/evaluation_report.txt"
        )
        print(report)
        
        # Print summary
        print("\nSUMMARY:")
        print("-" * 20)
        top1_acc = evaluation_results.get('top_1_accuracy', 0)
        top3_acc = evaluation_results.get('top_3_accuracy', 0)
        
        print(f"✅ Zero-shot Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"✅ Zero-shot Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        print(f"✅ Total test samples: {evaluation_results.get('total_samples', 'N/A')}")
        print(f"✅ Number of unseen languages: {len(config.UNSEEN_LANGUAGES)}")
        
        # Performance interpretation
        print("\nPERFORMANCE INTERPRETATION:")
        print("-" * 30)
        if top1_acc > 0.6:
            print("🎉 Excellent performance! The model generalizes very well to unseen languages.")
        elif top1_acc > 0.4:
            print("👍 Good performance! The model shows promising zero-shot capabilities.")
        elif top1_acc > 0.2:
            print("🤔 Moderate performance. The model has learned some transferable patterns.")
        else:
            print("⚠️  Low performance. Consider adjusting hyperparameters or model architecture.")
        
        if top3_acc > top1_acc + 0.2:
            print("📈 Significant improvement in Top-3 accuracy suggests the model is on the right track.")
        
        print()
        print("="*80)
        print(f"Execution completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    """Entry point for the script."""
    exit_code = main()
    sys.exit(exit_code)