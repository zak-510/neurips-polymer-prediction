"""Data loading and preprocessing utilities"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class PolymerDataset(Dataset):
    """PyTorch Dataset for polymer property prediction"""

    def __init__(
        self,
        smiles: List[str],
        targets: np.ndarray,
        tokenizer,
        max_length: int = 512,
        auxiliary_features: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset

        Args:
            smiles: List of SMILES strings
            targets: Target values (n_samples, n_tasks)
            tokenizer: SMILES tokenizer
            max_length: Maximum sequence length
            auxiliary_features: Optional auxiliary features
        """
        self.smiles = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.auxiliary_features = auxiliary_features

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index"""
        smiles = self.smiles[idx]

        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

        # Add auxiliary features if available
        if self.auxiliary_features is not None:
            item['auxiliary_features'] = torch.tensor(
                self.auxiliary_features[idx],
                dtype=torch.float32
            )

        return item


def load_competition_data(
    train_path: str,
    test_path: str,
    supplement_paths: Optional[List[str]] = None,
    target_columns: List[str] = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load competition data

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        supplement_paths: Paths to supplemental datasets
        target_columns: Target column names

    Returns:
        Tuple of (train_df, test_df, supplement_df)
    """
    # Load main training data
    train_df = pd.read_csv(train_path)

    # Load test data
    test_df = pd.read_csv(test_path)

    # Load supplemental data if provided
    supplement_df = None
    if supplement_paths:
        supplement_dfs = []
        for path in supplement_paths:
            df = pd.read_csv(path)
            supplement_dfs.append(df)
        supplement_df = pd.concat(supplement_dfs, ignore_index=True)

    return train_df, test_df, supplement_df


def prepare_data(
    df: pd.DataFrame,
    target_columns: List[str],
    smiles_column: str = 'SMILES'
) -> Tuple[List[str], np.ndarray]:
    """
    Prepare data for training

    Args:
        df: Input dataframe
        target_columns: Target column names
        smiles_column: SMILES column name

    Returns:
        Tuple of (smiles_list, targets_array)
    """
    smiles_list = df[smiles_column].tolist()

    # Extract targets (handle missing values)
    targets = []
    for col in target_columns:
        if col in df.columns:
            targets.append(df[col].values)
        else:
            # If column doesn't exist, fill with NaN
            targets.append(np.full(len(df), np.nan))

    targets_array = np.stack(targets, axis=1)  # (n_samples, n_tasks)

    return smiles_list, targets_array


def normalize_targets(
    targets: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize target values (per task)

    Args:
        targets: Target array (n_samples, n_tasks)
        scaler: Optional fitted scaler
        fit: Whether to fit scaler

    Returns:
        Tuple of (normalized_targets, scaler)
    """
    targets_normalized = targets.copy()
    n_tasks = targets.shape[1]

    if scaler is None:
        scaler = StandardScaler()

    for i in range(n_tasks):
        # Get valid (non-NaN) values for this task
        task_values = targets[:, i]
        valid_mask = ~np.isnan(task_values)

        if valid_mask.sum() > 0:
            valid_values = task_values[valid_mask].reshape(-1, 1)

            if fit:
                # Fit and transform
                normalized_values = scaler.fit_transform(valid_values)
            else:
                # Only transform
                normalized_values = scaler.transform(valid_values)

            # Put normalized values back
            targets_normalized[valid_mask, i] = normalized_values.flatten()

    return targets_normalized, scaler


def create_data_loaders(
    train_smiles: List[str],
    train_targets: np.ndarray,
    val_smiles: Optional[List[str]],
    val_targets: Optional[np.ndarray],
    tokenizer,
    batch_size: int = 32,
    train_auxiliary_features: Optional[np.ndarray] = None,
    val_auxiliary_features: Optional[np.ndarray] = None,
    max_length: int = 512,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch data loaders

    Args:
        train_smiles: Training SMILES
        train_targets: Training targets
        val_smiles: Validation SMILES
        val_targets: Validation targets
        tokenizer: SMILES tokenizer
        batch_size: Batch size
        train_auxiliary_features: Training auxiliary features
        val_auxiliary_features: Validation auxiliary features
        max_length: Maximum sequence length
        num_workers: Number of data loader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = PolymerDataset(
        smiles=train_smiles,
        targets=train_targets,
        tokenizer=tokenizer,
        max_length=max_length,
        auxiliary_features=train_auxiliary_features
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create validation dataset if provided
    val_loader = None
    if val_smiles is not None and val_targets is not None:
        val_dataset = PolymerDataset(
            smiles=val_smiles,
            targets=val_targets,
            tokenizer=tokenizer,
            max_length=max_length,
            auxiliary_features=val_auxiliary_features
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def create_kfold_splits(
    smiles: List[str],
    targets: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create K-fold cross-validation splits

    Args:
        smiles: List of SMILES
        targets: Target array
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) tuples
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for train_idx, val_idx in kfold.split(smiles):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits
