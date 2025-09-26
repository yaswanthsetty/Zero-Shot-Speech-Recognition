"""
Evaluation module for Zero-Shot Spoken Language Identification.

This module contains functions for evaluating the trained model on unseen
languages using cosine similarity and computing top-k accuracy metrics.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from . import config
from .model import ProjectionHead
from .features import compute_cosine_similarity, normalize_embeddings


def evaluate_zero_shot(
    model: ProjectionHead,
    test_loader: DataLoader,
    all_language_vectors: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate the model on zero-shot language identification.
    
    This function evaluates the trained model on unseen languages by:
    1. Extracting embeddings for test audio samples
    2. Computing cosine similarity with all language phonological vectors
    3. Ranking languages by similarity and computing top-k accuracy
    
    Args:
        model: Trained ProjectionHead model
        test_loader: DataLoader containing unseen language data
        all_language_vectors: Dictionary of all language phonological vectors
        device: Device to use for evaluation
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = config.DEVICE
    
    print("Starting zero-shot evaluation...")
    print(f"Number of languages in phonological space: {len(all_language_vectors)}")
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Move all language vectors to device and create matrix
    lang_codes = list(all_language_vectors.keys())
    lang_vectors_list = [all_language_vectors[lang].to(device) for lang in lang_codes]
    lang_matrix = torch.stack(lang_vectors_list)  # [num_languages, embedding_dim]
    
    if verbose:
        print(f"Language matrix shape: {lang_matrix.shape}")
        print(f"Available languages: {lang_codes}")
    
    # Storage for predictions and ground truth
    all_predictions = []
    all_true_languages = []
    all_similarities = []
    
    # Evaluate on test data
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating") if verbose else test_loader
        
        for batch in pbar:
            # Get batch data
            audio_features = batch['audio_features'].to(device)
            true_languages = batch['languages']
            
            # Get model predictions
            predicted_embeddings = model(audio_features)  # [batch_size, embedding_dim]
            
            # Compute similarities with all languages
            similarities = compute_cosine_similarity(
                predicted_embeddings, lang_matrix
            )  # [batch_size, num_languages]
            
            # Store results
            all_predictions.append(predicted_embeddings.cpu())
            all_similarities.append(similarities.cpu())
            all_true_languages.extend(true_languages)
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_similarities = torch.cat(all_similarities, dim=0)
    
    print(f"Total test samples: {len(all_true_languages)}")
    
    # Compute top-k accuracies
    results = {}
    for k in config.TOP_K_VALUES:
        accuracy = compute_topk_accuracy(
            all_similarities, all_true_languages, lang_codes, k
        )
        results[f'top_{k}_accuracy'] = accuracy
        if verbose:
            print(f"Top-{k} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compute per-language results
    per_language_results = compute_per_language_accuracy(
        all_similarities, all_true_languages, lang_codes
    )
    
    if verbose:
        print("\nPer-language Top-1 Accuracy:")
        for lang, acc in sorted(per_language_results.items()):
            print(f"  {lang}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Compute confusion matrix (top predictions for each true language)
    confusion_info = compute_confusion_analysis(
        all_similarities, all_true_languages, lang_codes, top_k=3
    )
    
    if verbose:
        print("\nConfusion Analysis (Top-3 predictions for each language):")
        for true_lang, predictions in confusion_info.items():
            pred_str = ", ".join([f"{lang}({score:.3f})" for lang, score in predictions[:3]])
            print(f"  {true_lang} -> {pred_str}")
    
    # Add additional metrics
    results['per_language_accuracy'] = per_language_results
    results['confusion_analysis'] = confusion_info
    results['total_samples'] = len(all_true_languages)
    results['num_languages'] = len(lang_codes)
    
    return results


def compute_topk_accuracy(
    similarities: torch.Tensor,
    true_languages: List[str],
    lang_codes: List[str],
    k: int
) -> float:
    """
    Compute top-k accuracy from similarity scores.
    
    Args:
        similarities: Similarity matrix [num_samples, num_languages]
        true_languages: List of true language labels
        lang_codes: List of language codes corresponding to similarity columns
        k: K for top-k accuracy
        
    Returns:
        Top-k accuracy
    """
    # Get top-k predictions
    _, top_k_indices = torch.topk(similarities, k=k, dim=1)
    
    correct = 0
    for i, true_lang in enumerate(true_languages):
        # Get predicted languages for this sample
        predicted_indices = top_k_indices[i]
        predicted_languages = [lang_codes[idx] for idx in predicted_indices]
        
        # Check if true language is in top-k predictions
        if true_lang in predicted_languages:
            correct += 1
    
    accuracy = correct / len(true_languages)
    return accuracy


def compute_per_language_accuracy(
    similarities: torch.Tensor,
    true_languages: List[str],
    lang_codes: List[str]
) -> Dict[str, float]:
    """
    Compute top-1 accuracy for each language separately.
    
    Args:
        similarities: Similarity matrix [num_samples, num_languages]
        true_languages: List of true language labels
        lang_codes: List of language codes
        
    Returns:
        Dictionary mapping language codes to their accuracy
    """
    # Group samples by language
    language_indices = defaultdict(list)
    for i, lang in enumerate(true_languages):
        language_indices[lang].append(i)
    
    per_lang_accuracy = {}
    
    for lang, indices in language_indices.items():
        if not indices:
            continue
            
        # Get similarities for this language
        lang_similarities = similarities[indices]
        
        # Get top-1 predictions
        _, top1_indices = torch.topk(lang_similarities, k=1, dim=1)
        
        # Count correct predictions
        correct = 0
        for i, pred_idx in enumerate(top1_indices.squeeze()):
            predicted_lang = lang_codes[pred_idx]
            if predicted_lang == lang:
                correct += 1
        
        accuracy = correct / len(indices)
        per_lang_accuracy[lang] = accuracy
    
    return per_lang_accuracy


def compute_confusion_analysis(
    similarities: torch.Tensor,
    true_languages: List[str],
    lang_codes: List[str],
    top_k: int = 3
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute confusion analysis showing top predictions for each true language.
    
    Args:
        similarities: Similarity matrix [num_samples, num_languages]
        true_languages: List of true language labels
        lang_codes: List of language codes
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary mapping true languages to their top predicted languages with scores
    """
    # Group samples by language
    language_indices = defaultdict(list)
    for i, lang in enumerate(true_languages):
        language_indices[lang].append(i)
    
    confusion_analysis = {}
    
    for lang, indices in language_indices.items():
        if not indices:
            continue
        
        # Get similarities for this language
        lang_similarities = similarities[indices]
        
        # Average similarities across all samples of this language
        avg_similarities = torch.mean(lang_similarities, dim=0)
        
        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(avg_similarities, k=top_k)
        
        # Convert to language codes with scores
        predictions = []
        for score, idx in zip(top_k_scores, top_k_indices):
            predicted_lang = lang_codes[idx]
            predictions.append((predicted_lang, score.item()))
        
        confusion_analysis[lang] = predictions
    
    return confusion_analysis


def analyze_embedding_space(
    model: ProjectionHead,
    test_loader: DataLoader,
    all_language_vectors: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None
) -> Dict[str, any]:
    """
    Analyze the learned embedding space to understand model behavior.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        all_language_vectors: Language phonological vectors
        device: Device to use
        
    Returns:
        Dictionary containing analysis results
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    model = model.to(device)
    
    # Collect embeddings by language
    language_embeddings = defaultdict(list)
    
    with torch.no_grad():
        for batch in test_loader:
            audio_features = batch['audio_features'].to(device)
            languages = batch['languages']
            
            predicted_embeddings = model(audio_features)
            
            for i, lang in enumerate(languages):
                language_embeddings[lang].append(predicted_embeddings[i].cpu())
    
    # Compute statistics for each language
    analysis_results = {}
    
    for lang, embeddings in language_embeddings.items():
        if not embeddings:
            continue
            
        embeddings_tensor = torch.stack(embeddings)
        
        # Compute statistics
        mean_embedding = torch.mean(embeddings_tensor, dim=0)
        std_embedding = torch.std(embeddings_tensor, dim=0)
        
        # Compute similarity to target phonological vector
        if lang in all_language_vectors:
            target_vector = all_language_vectors[lang]
            similarity_to_target = F.cosine_similarity(
                mean_embedding.unsqueeze(0), 
                target_vector.unsqueeze(0)
            ).item()
        else:
            similarity_to_target = 0.0
        
        analysis_results[lang] = {
            'num_samples': len(embeddings),
            'mean_embedding': mean_embedding,
            'std_embedding': std_embedding,
            'similarity_to_target': similarity_to_target,
            'embedding_norm': torch.norm(mean_embedding).item()
        }
    
    return analysis_results


def generate_evaluation_report(
    results: Dict[str, any],
    save_path: str = None
) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Results from evaluate_zero_shot
        save_path: Optional path to save the report
        
    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ZERO-SHOT LANGUAGE IDENTIFICATION EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("OVERALL PERFORMANCE:")
    report_lines.append("-" * 20)
    for metric, value in results.items():
        if 'accuracy' in metric and not isinstance(value, dict):
            report_lines.append(f"{metric.replace('_', ' ').title()}: {value:.4f} ({value*100:.2f}%)")
    report_lines.append("")
    
    # Dataset info
    report_lines.append("DATASET INFORMATION:")  
    report_lines.append("-" * 20)
    report_lines.append(f"Total test samples: {results.get('total_samples', 'N/A')}")
    report_lines.append(f"Number of languages: {results.get('num_languages', 'N/A')}")
    report_lines.append("")
    
    # Per-language results (if available)
    if 'per_language_accuracy' in results:
        report_lines.append("PER-LANGUAGE ACCURACY:")
        report_lines.append("-" * 20)
        per_lang = results['per_language_accuracy']
        for lang in sorted(per_lang.keys()):
            acc = per_lang[lang]
            report_lines.append(f"{lang}: {acc:.4f} ({acc*100:.2f}%)")
        report_lines.append("")
    
    # Best and worst performing languages
    if 'per_language_accuracy' in results:
        per_lang = results['per_language_accuracy']
        sorted_langs = sorted(per_lang.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append("BEST PERFORMING LANGUAGES:")
        report_lines.append("-" * 20)
        for lang, acc in sorted_langs[:3]:
            report_lines.append(f"{lang}: {acc:.4f} ({acc*100:.2f}%)")
        report_lines.append("")
        
        report_lines.append("WORST PERFORMING LANGUAGES:")
        report_lines.append("-" * 20)
        for lang, acc in sorted_langs[-3:]:
            report_lines.append(f"{lang}: {acc:.4f} ({acc*100:.2f}%)")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Evaluation report saved to: {save_path}")
    
    return report