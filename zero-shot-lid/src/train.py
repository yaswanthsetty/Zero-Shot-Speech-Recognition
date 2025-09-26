"""
Training module for Zero-Shot Spoken Language Identification.

This module contains the training logic for the ProjectionHead model,
including optimization, loss computation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os

from . import config
from .model import ProjectionHead
from .features import normalize_embeddings, compute_cosine_similarity


def train_model(
    model: ProjectionHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phonological_vectors: Dict[str, torch.Tensor],
    num_epochs: int = None,
    learning_rate: float = None,
    device: Optional[torch.device] = None
) -> ProjectionHead:
    """
    Train the ProjectionHead model to map audio embeddings to phonological space.
    
    Args:
        model: ProjectionHead model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        phonological_vectors: Dictionary mapping language codes to phonological vectors
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to use for training
        
    Returns:
        Trained model
    """
    # Use config defaults if not specified
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if device is None:
        device = config.DEVICE
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Initialize loss function (Cosine Embedding Loss)
    criterion = nn.CosineEmbeddingLoss(margin=0.1, reduction='mean')
    
    # Move phonological vectors to device
    phonological_vectors_tensor = {}
    for lang, vector in phonological_vectors.items():
        phonological_vectors_tensor[lang] = vector.to(device)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = train_epoch(
            model, train_loader, phonological_vectors_tensor, 
            criterion, optimizer, device
        )
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = validate_epoch(
            model, val_loader, phonological_vectors_tensor,
            criterion, device
        )
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling (optional)
        if epoch > 0 and epoch % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
            print(f"Learning rate reduced to: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save the trained model
    print(f"\nSaving model to {config.MODEL_SAVE_PATH}")
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': config.BATCH_SIZE,
            'audio_embedding_dim': config.AUDIO_EMBEDDING_DIM,
            'phonological_embedding_dim': config.PHONOLOGICAL_EMBEDDING_DIM
        }
    }, config.MODEL_SAVE_PATH)
    
    print("Training completed successfully!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return model


def train_epoch(
    model: ProjectionHead,
    train_loader: DataLoader,
    phonological_vectors: Dict[str, torch.Tensor],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        phonological_vectors: Phonological vectors for languages
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(pbar):
        # Get batch data
        audio_features = batch['audio_features'].to(device)
        languages = batch['languages']
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predicted_embeddings = model(audio_features)
        
        # Get target phonological vectors for this batch
        target_embeddings = []
        for lang in languages:
            if lang in phonological_vectors:
                target_embeddings.append(phonological_vectors[lang])
            else:
                # Fallback to a default vector if language not found
                print(f"Warning: Language {lang} not found in phonological vectors")
                # Use the first available language vector as fallback
                fallback_lang = list(phonological_vectors.keys())[0]
                target_embeddings.append(phonological_vectors[fallback_lang])
        
        target_embeddings = torch.stack(target_embeddings).to(device)
        
        # Create target labels (all positive pairs since we want similarity)
        targets = torch.ones(audio_features.size(0)).to(device)
        
        # Compute loss
        loss = criterion(predicted_embeddings, target_embeddings, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log progress occasionally
        if config.VERBOSE and batch_idx % config.LOG_INTERVAL == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate_epoch(
    model: ProjectionHead,
    val_loader: DataLoader,
    phonological_vectors: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        phonological_vectors: Phonological vectors for languages
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch in pbar:
            # Get batch data
            audio_features = batch['audio_features'].to(device)
            languages = batch['languages']
            
            # Forward pass
            predicted_embeddings = model(audio_features)
            
            # Get target phonological vectors
            target_embeddings = []
            for lang in languages:
                if lang in phonological_vectors:
                    target_embeddings.append(phonological_vectors[lang])
                else:
                    # Fallback to a default vector
                    fallback_lang = list(phonological_vectors.keys())[0]
                    target_embeddings.append(phonological_vectors[fallback_lang])
            
            target_embeddings = torch.stack(target_embeddings).to(device)
            
            # Create target labels
            targets = torch.ones(audio_features.size(0)).to(device)
            
            # Compute loss
            loss = criterion(predicted_embeddings, target_embeddings, targets)
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def compute_accuracy(
    predicted_embeddings: torch.Tensor,
    target_language_vectors: Dict[str, torch.Tensor],
    true_languages: List[str],
    top_k: int = 1
) -> float:
    """
    Compute top-k accuracy for language identification.
    
    Args:
        predicted_embeddings: Predicted embeddings from the model
        target_language_vectors: All language phonological vectors
        true_languages: True language labels
        top_k: K for top-k accuracy
        
    Returns:
        Top-k accuracy
    """
    # Create matrix of all language vectors
    lang_codes = list(target_language_vectors.keys())
    lang_matrix = torch.stack([target_language_vectors[lang] for lang in lang_codes])
    
    # Compute similarities
    similarities = compute_cosine_similarity(predicted_embeddings, lang_matrix)
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
    
    # Convert indices to language codes
    correct = 0
    for i, true_lang in enumerate(true_languages):
        predicted_langs = [lang_codes[idx] for idx in top_k_indices[i]]
        if true_lang in predicted_langs:
            correct += 1
    
    accuracy = correct / len(true_languages)
    return accuracy


def evaluate_during_training(
    model: ProjectionHead,
    val_loader: DataLoader,
    phonological_vectors: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model accuracy during training.
    
    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        phonological_vectors: Phonological vectors for all languages
        device: Device to use
        
    Returns:
        Tuple of (top-1 accuracy, top-3 accuracy)
    """
    model.eval()
    all_predictions = []
    all_true_languages = []
    
    with torch.no_grad():
        for batch in val_loader:
            audio_features = batch['audio_features'].to(device)
            languages = batch['languages']
            
            # Get predictions
            predicted_embeddings = model(audio_features)
            
            all_predictions.append(predicted_embeddings.cpu())
            all_true_languages.extend(languages)
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Move phonological vectors to CPU for evaluation
    cpu_phonological_vectors = {k: v.cpu() for k, v in phonological_vectors.items()}
    
    # Compute accuracies
    top1_acc = compute_accuracy(all_predictions, cpu_phonological_vectors, all_true_languages, top_k=1)
    top3_acc = compute_accuracy(all_predictions, cpu_phonological_vectors, all_true_languages, top_k=3)
    
    return top1_acc, top3_acc