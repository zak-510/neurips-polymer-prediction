"""Training script for ChemBERTa multi-task model"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from models.chemberta_multitask import ChemBERTaMultiTask, WeightedMSELoss, load_chemberta_tokenizer
from features.molecular_features import MolecularFeatureExtractor
from utils.data_loader import (
    load_competition_data,
    prepare_data,
    normalize_targets,
    create_data_loaders,
    create_kfold_splits
)
from utils.metrics import compute_multitask_metrics, print_metrics
from utils.config import Config


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        auxiliary_features = None
        if 'auxiliary_features' in batch:
            auxiliary_features = batch['auxiliary_features'].to(device)

        # Forward pass
        predictions = model(input_ids, attention_mask, auxiliary_features)

        # Compute loss
        mask = ~torch.isnan(targets)
        loss = criterion(predictions, targets, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device, target_names):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            auxiliary_features = None
            if 'auxiliary_features' in batch:
                auxiliary_features = batch['auxiliary_features'].to(device)

            # Forward pass
            predictions = model(input_ids, attention_mask, auxiliary_features)

            # Compute loss
            mask = ~torch.isnan(targets)
            loss = criterion(predictions, targets, mask)

            total_loss += loss.item()
            num_batches += 1

            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Compute metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_multitask_metrics(all_targets, all_predictions, target_names)
    metrics['val_loss'] = avg_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train ChemBERTa multi-task model')
    parser.add_argument('--config', type=str, default='configs/chemberta_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number for cross-validation (None = train on all data)')
    args = parser.parse_args()

    # Load config
    config = Config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed
    seed = config['data'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device (prioritize MPS for Apple Silicon, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: mps (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: cuda (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: cpu (Warning: Training will be slow)")

    # Set number of workers for DataLoader (0 for MPS to avoid multiprocessing issues)
    num_workers = 0 if device.type == 'mps' else 4

    # Create output directory
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df, test_df, supplement_df = load_competition_data(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path'],
        supplement_paths=config['data'].get('supplement_paths', None),
        target_columns=config['data']['targets']
    )

    # Prepare data
    train_smiles, train_targets = prepare_data(
        train_df,
        target_columns=config['data']['targets']
    )

    print(f"Training samples: {len(train_smiles)}")
    print(f"Target columns: {config['data']['targets']}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_chemberta_tokenizer(config['model']['pretrained_model'])

    # Extract molecular features if configured
    train_auxiliary_features = None
    val_auxiliary_features = None

    if config['features'].get('use_rdkit_features', False):
        print("Extracting molecular features...")
        feature_extractor = MolecularFeatureExtractor(
            use_descriptors=True,
            use_fingerprints=False
        )
        train_auxiliary_features = feature_extractor.fit_transform(train_smiles)

    # Create K-fold splits or single train/val split
    if args.fold is not None:
        print(f"Using fold {args.fold} for cross-validation...")
        n_folds = config['data'].get('n_folds', 5)
        splits = create_kfold_splits(train_smiles, train_targets, n_folds=n_folds, seed=seed)
        train_idx, val_idx = splits[args.fold]

        fold_train_smiles = [train_smiles[i] for i in train_idx]
        fold_val_smiles = [train_smiles[i] for i in val_idx]
        fold_train_targets = train_targets[train_idx]
        fold_val_targets = train_targets[val_idx]

        if train_auxiliary_features is not None:
            fold_train_aux = train_auxiliary_features[train_idx]
            fold_val_aux = train_auxiliary_features[val_idx]
        else:
            fold_train_aux = None
            fold_val_aux = None
    else:
        print("Using single train/val split...")
        val_split = config['data'].get('val_split', 0.2)
        n_val = int(len(train_smiles) * val_split)

        fold_train_smiles = train_smiles[n_val:]
        fold_val_smiles = train_smiles[:n_val]
        fold_train_targets = train_targets[n_val:]
        fold_val_targets = train_targets[:n_val]

        if train_auxiliary_features is not None:
            fold_train_aux = train_auxiliary_features[n_val:]
            fold_val_aux = train_auxiliary_features[:n_val]
        else:
            fold_train_aux = None
            fold_val_aux = None

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_smiles=fold_train_smiles,
        train_targets=fold_train_targets,
        val_smiles=fold_val_smiles,
        val_targets=fold_val_targets,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        train_auxiliary_features=fold_train_aux,
        val_auxiliary_features=fold_val_aux,
        num_workers=num_workers
    )

    # Create model
    print("Creating model...")
    model = ChemBERTaMultiTask(
        pretrained_model=config['model']['pretrained_model'],
        num_tasks=config['model']['num_tasks'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        freeze_encoder=False,
        use_auxiliary_features=config['features'].get('use_rdkit_features', False),
        auxiliary_feature_dim=train_auxiliary_features.shape[1] if train_auxiliary_features is not None else 0
    )
    model = model.to(device)

    # Create loss function
    task_weights = [config['loss']['weights'][name] for name in config['data']['targets']]
    criterion = WeightedMSELoss(task_weights=task_weights)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training'].get('patience', 10)

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Unfreeze encoder after N epochs
        freeze_epochs = config['model'].get('freeze_encoder_epochs', 0)
        if epoch == freeze_epochs and freeze_epochs > 0:
            print("Unfreezing encoder...")
            model.unfreeze_encoder()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        metrics = validate(model, val_loader, criterion, device, config['data']['targets'])
        print_metrics(metrics, prefix="Validation Metrics")

        # Save best model
        val_loss = metrics['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            model_path = output_dir / f'best_model_fold{args.fold if args.fold is not None else 0}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, model_path)
            print(f"Saved best model to {model_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
