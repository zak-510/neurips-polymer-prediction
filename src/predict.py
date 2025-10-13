"""Inference script for generating predictions"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from models.chemberta_multitask import ChemBERTaMultiTask, load_chemberta_tokenizer
from features.molecular_features import MolecularFeatureExtractor
from utils.data_loader import load_competition_data, prepare_data, PolymerDataset
from utils.config import Config
from torch.utils.data import DataLoader


def predict(model, data_loader, device):
    """Generate predictions"""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            auxiliary_features = None
            if 'auxiliary_features' in batch:
                auxiliary_features = batch['auxiliary_features'].to(device)

            predictions = model(input_ids, attention_mask, auxiliary_features)
            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--config', type=str, default='configs/chemberta_baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='submissions/',
                        help='Output directory for predictions')
    args = parser.parse_args()

    # Load config
    config = Config(args.config)
    print(f"Loaded config from {args.config}")

    # Set device (prioritize MPS for Apple Silicon, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: mps (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: cuda (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: cpu")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading test data...")
    _, test_df, _ = load_competition_data(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path'],
        target_columns=config['data']['targets']
    )

    test_smiles = test_df['SMILES'].tolist()
    test_ids = test_df['id'].tolist()

    print(f"Test samples: {len(test_smiles)}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_chemberta_tokenizer(config['model']['pretrained_model'])

    # Extract molecular features if needed
    test_auxiliary_features = None
    if config['features'].get('use_rdkit_features', False):
        print("Extracting molecular features...")
        feature_extractor = MolecularFeatureExtractor(
            use_descriptors=True,
            use_fingerprints=False
        )
        test_auxiliary_features = feature_extractor.fit_transform(test_smiles)

    # Create dummy targets for dataset
    dummy_targets = np.zeros((len(test_smiles), config['model']['num_tasks']))

    # Create dataset and loader
    test_dataset = PolymerDataset(
        smiles=test_smiles,
        targets=dummy_targets,
        tokenizer=tokenizer,
        auxiliary_features=test_auxiliary_features
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = ChemBERTaMultiTask(
        pretrained_model=config['model']['pretrained_model'],
        num_tasks=config['model']['num_tasks'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        use_auxiliary_features=config['features'].get('use_rdkit_features', False),
        auxiliary_feature_dim=test_auxiliary_features.shape[1] if test_auxiliary_features is not None else 0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Generate predictions
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_ids
    })

    for i, target_name in enumerate(config['data']['targets']):
        submission_df[target_name] = predictions[:, i]

    # Save submission
    submission_path = output_dir / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    print("\nPrediction Summary:")
    print(submission_df)


if __name__ == '__main__':
    main()
