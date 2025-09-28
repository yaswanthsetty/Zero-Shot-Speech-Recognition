"""
Model definition for Zero-Shot Spoken Language Identification.

This module contains the ProjectionHead model that maps audio embeddings
to the phonological embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config


class ProjectionHead(nn.Module):
    """
    Multi-layer perceptron that projects audio embeddings to phonological space.
    
    This model learns to map high-dimensional audio embeddings from a pre-trained
    model (e.g., Wav2Vec2) into the lower-dimensional phonological feature space
    where languages can be identified based on their phonological properties.
    
    Architecture:
        - Input layer: AUDIO_EMBEDDING_DIM -> HIDDEN_DIM
        - Hidden layer(s): HIDDEN_DIM -> HIDDEN_DIM (with ReLU and Dropout)
        - Output layer: HIDDEN_DIM -> PHONOLOGICAL_EMBEDDING_DIM
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = None,
        dropout_rate: float = None,
        num_hidden_layers: int = 2
    ):
        """
        Initialize the ProjectionHead model.
        
        Args:
            input_dim: Input dimension (audio embedding size)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (phonological embedding size)
            dropout_rate: Dropout probability
            num_hidden_layers: Number of hidden layers
        """
        super(ProjectionHead, self).__init__()
        
        # Use config defaults if not specified
        self.input_dim = input_dim or config.AUDIO_EMBEDDING_DIM
        self.hidden_dim = hidden_dim or config.HIDDEN_DIM
        self.output_dim = output_dim or config.PHONOLOGICAL_EMBEDDING_DIM
        self.dropout_rate = dropout_rate or config.DROPOUT_RATE
        self.num_hidden_layers = num_hidden_layers
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer (no activation, as we'll use cosine similarity)
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        # Create the sequential model
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            audio_embeddings: Input audio embeddings [batch_size, input_dim]
            
        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        # Ensure input has correct shape
        if audio_embeddings.dim() == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)
        
        # Project to phonological space
        projected = self.projection(audio_embeddings)
        
        return projected
    
    def get_model_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'num_hidden_layers': self.num_hidden_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning language embeddings.
    
    This loss function encourages audio embeddings from the same language
    to be close to their target phonological vectors, while pushing
    embeddings from different languages apart.
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize the contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            temperature: Temperature parameter for scaling similarities
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            predicted_embeddings: Predicted embeddings from the model [N, D]
            target_embeddings: Target phonological embeddings [N, D]
            labels: Language labels for computing positive/negative pairs [N]
            
        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        pred_norm = F.normalize(predicted_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = torch.mm(pred_norm, target_norm.t()) / self.temperature
        
        # Create positive mask (same language)
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        
        # Create negative mask
        negative_mask = 1.0 - positive_mask
        
        # Compute positive loss (maximize similarity for same language)
        positive_loss = -torch.sum(similarities * positive_mask) / torch.sum(positive_mask)
        
        # Compute negative loss (minimize similarity for different languages)
        negative_similarities = similarities * negative_mask
        negative_loss = torch.sum(torch.relu(negative_similarities + self.margin)) / torch.sum(negative_mask)
        
        total_loss = positive_loss + negative_loss
        
        return total_loss


def create_model(device: Optional[torch.device] = None) -> ProjectionHead:
    """
    Create and initialize a ProjectionHead model.
    
    Args:
        device: Device to place the model on
        
    Returns:
        Initialized ProjectionHead model
    """
    if device is None:
        device = config.DEVICE
    
    model = ProjectionHead()
    model = model.to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Created ProjectionHead model:")
    print(f"  - Input dimension: {model_info['input_dim']}")
    print(f"  - Hidden dimension: {model_info['hidden_dim']}")
    print(f"  - Output dimension: {model_info['output_dim']}")
    print(f"  - Hidden layers: {model_info['num_hidden_layers']}")
    print(f"  - Dropout rate: {model_info['dropout_rate']}")
    print(f"  - Total parameters: {model_info['total_parameters']:,}")
    print(f"  - Device: {device}")
    
    return model


def save_model(model: ProjectionHead, path: str):
    """
    Save model state dict to file.
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'config': {
            'audio_embedding_dim': config.AUDIO_EMBEDDING_DIM,
            'phonological_embedding_dim': config.PHONOLOGICAL_EMBEDDING_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'dropout_rate': config.DROPOUT_RATE
        }
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: Optional[torch.device] = None) -> ProjectionHead:
    """
    Load model from saved state dict.
    
    Args:
        path: Path to the saved model
        device: Device to place the model on
        
    Returns:
        Loaded ProjectionHead model
    """
    if device is None:
        device = config.DEVICE
    
    checkpoint = torch.load(path, map_location=device)
    
    # Create model with saved configuration
    saved_config = checkpoint.get('config', {})
    model = ProjectionHead(
        input_dim=saved_config.get('audio_embedding_dim', config.AUDIO_EMBEDDING_DIM),
        hidden_dim=saved_config.get('hidden_dim', config.HIDDEN_DIM),
        output_dim=saved_config.get('phonological_embedding_dim', config.PHONOLOGICAL_EMBEDDING_DIM),
        dropout_rate=saved_config.get('dropout_rate', config.DROPOUT_RATE)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model