"""SMILES preprocessing and augmentation utilities"""

import random
from typing import List, Optional
import numpy as np


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string using RDKit

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def enumerate_smiles(smiles: str, n_variants: int = 5) -> List[str]:
    """
    Generate multiple valid SMILES representations (SMILES augmentation)

    Args:
        smiles: Input SMILES string
        n_variants: Number of variants to generate

    Returns:
        List of SMILES variants
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        variants = set()
        variants.add(smiles)  # Include original

        # Generate randomized SMILES
        for _ in range(n_variants * 3):  # Try more times to get n_variants
            randomized = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(randomized)
            if len(variants) >= n_variants + 1:  # +1 for original
                break

        return list(variants)[:n_variants]
    except:
        return [smiles]


def validate_smiles(smiles: str) -> bool:
    """
    Check if SMILES string is valid

    Args:
        smiles: Input SMILES string

    Returns:
        True if valid, False otherwise
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def clean_smiles(smiles: str) -> str:
    """
    Clean SMILES string (remove salts, standardize)

    Args:
        smiles: Input SMILES string

    Returns:
        Cleaned SMILES string
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import SaltRemover

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        # Remove salts
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        # Standardize
        cleaned = Chem.MolToSmiles(mol, canonical=True)
        return cleaned
    except:
        return smiles


def get_polymer_features(smiles: str) -> dict:
    """
    Extract polymer-specific features from SMILES

    Args:
        smiles: Polymer SMILES string

    Returns:
        Dictionary of polymer features
    """
    features = {
        'has_repeating_unit': '*' in smiles,
        'num_stars': smiles.count('*'),
        'smiles_length': len(smiles),
        'num_rings': smiles.count('1') + smiles.count('2'),  # Simplified ring count
        'num_double_bonds': smiles.count('='),
        'num_triple_bonds': smiles.count('#'),
        'num_aromatic': smiles.count('c') + smiles.count('n') + smiles.count('o') + smiles.count('s'),
    }

    return features


class SMILESTokenizer:
    """Custom SMILES tokenizer compatible with transformers"""

    def __init__(self, pretrained_tokenizer):
        """
        Initialize with pretrained tokenizer

        Args:
            pretrained_tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = pretrained_tokenizer

    def __call__(
        self,
        smiles: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: Optional[str] = None
    ):
        """
        Tokenize SMILES strings

        Args:
            smiles: List of SMILES strings
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return type ('pt' for PyTorch)

        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            smiles,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )

    def batch_encode(self, smiles: List[str], **kwargs):
        """Batch encode SMILES strings"""
        return self(smiles, **kwargs)


def augment_dataset_with_smiles(
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
    n_augmentations: int = 3
) -> tuple:
    """
    Augment dataset with SMILES enumeration

    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels array (n_samples, n_tasks)
        n_augmentations: Number of augmentations per sample

    Returns:
        Tuple of (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for i, smiles in enumerate(smiles_list):
        # Add original
        augmented_smiles.append(smiles)
        if labels is not None:
            augmented_labels.append(labels[i])

        # Add augmented versions
        variants = enumerate_smiles(smiles, n_variants=n_augmentations)
        for variant in variants[1:]:  # Skip first (original)
            augmented_smiles.append(variant)
            if labels is not None:
                augmented_labels.append(labels[i])

    if labels is not None:
        augmented_labels = np.array(augmented_labels)

    return augmented_smiles, augmented_labels
