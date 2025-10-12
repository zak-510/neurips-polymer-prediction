"""ChemBERTa Multi-Task Regression Model"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional


class ChemBERTaMultiTask(nn.Module):
    """
    ChemBERTa-based multi-task regression model for polymer property prediction
    """

    def __init__(
        self,
        pretrained_model: str = "seyonec/ChemBERTa-zinc-base-v1",
        num_tasks: int = 5,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        use_auxiliary_features: bool = False,
        auxiliary_feature_dim: int = 0
    ):
        """
        Initialize ChemBERTa multi-task model

        Args:
            pretrained_model: HuggingFace model name
            num_tasks: Number of prediction tasks
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            freeze_encoder: Whether to freeze encoder weights
            use_auxiliary_features: Whether to use additional molecular features
            auxiliary_feature_dim: Dimension of auxiliary features
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.use_auxiliary_features = use_auxiliary_features

        # Load pretrained ChemBERTa
        self.encoder = AutoModel.from_pretrained(pretrained_model)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size

        # Projection layer
        projection_input_dim = self.hidden_size
        if use_auxiliary_features:
            projection_input_dim += auxiliary_feature_dim

        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_tasks)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        auxiliary_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            auxiliary_features: Additional features (batch_size, aux_dim)

        Returns:
            Predictions (batch_size, num_tasks)
        """
        # Encode SMILES
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Concatenate auxiliary features if provided
        if self.use_auxiliary_features and auxiliary_features is not None:
            pooled_output = torch.cat([pooled_output, auxiliary_features], dim=1)

        # Project to shared representation
        shared_repr = self.projection(pooled_output)  # (batch_size, hidden_dim)

        # Task-specific predictions
        task_outputs = []
        for head in self.task_heads:
            task_output = head(shared_repr)  # (batch_size, 1)
            task_outputs.append(task_output)

        # Concatenate all task outputs
        predictions = torch.cat(task_outputs, dim=1)  # (batch_size, num_tasks)

        return predictions

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss that handles missing values"""

    def __init__(self, task_weights: Optional[List[float]] = None):
        """
        Initialize weighted MSE loss

        Args:
            task_weights: Optional weights for each task
        """
        super().__init__()
        self.task_weights = task_weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss

        Args:
            predictions: Predicted values (batch_size, num_tasks)
            targets: Target values (batch_size, num_tasks)
            mask: Valid value mask (batch_size, num_tasks)

        Returns:
            Loss value
        """
        if mask is None:
            # Create mask from non-NaN values
            mask = ~torch.isnan(targets)

        # Replace NaN with 0 (they'll be masked out)
        targets = torch.nan_to_num(targets, nan=0.0)

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Apply mask
        masked_errors = squared_errors * mask.float()

        # Apply task weights if provided
        if self.task_weights is not None:
            weights = torch.tensor(self.task_weights, device=predictions.device)
            masked_errors = masked_errors * weights.unsqueeze(0)

        # Compute mean over valid values
        num_valid = mask.float().sum()
        if num_valid > 0:
            loss = masked_errors.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=predictions.device)

        return loss


def load_chemberta_tokenizer(pretrained_model: str = "seyonec/ChemBERTa-zinc-base-v1"):
    """
    Load ChemBERTa tokenizer

    Args:
        pretrained_model: HuggingFace model name

    Returns:
        Tokenizer
    """
    return AutoTokenizer.from_pretrained(pretrained_model)


def create_model(config: dict) -> ChemBERTaMultiTask:
    """
    Create ChemBERTa model from config

    Args:
        config: Model configuration dictionary

    Returns:
        ChemBERTaMultiTask model
    """
    model_config = config.get('model', {})

    model = ChemBERTaMultiTask(
        pretrained_model=model_config.get('pretrained_model', 'seyonec/ChemBERTa-zinc-base-v1'),
        num_tasks=model_config.get('num_tasks', 5),
        hidden_dim=model_config.get('hidden_dim', 768),
        dropout=model_config.get('dropout', 0.1),
        freeze_encoder=model_config.get('freeze_encoder', False),
        use_auxiliary_features=config.get('features', {}).get('use_rdkit_features', False),
        auxiliary_feature_dim=25  # Number of RDKit descriptors
    )

    return model
