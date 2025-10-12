"""Molecular descriptor computation using RDKit"""

import numpy as np
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


def compute_rdkit_descriptors(smiles: str) -> dict:
    """
    Compute RDKit molecular descriptors

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of descriptors
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, Crippen, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = {
            # Basic properties
            'MolWt': Descriptors.MolWt(mol),
            'ExactMolWt': Descriptors.ExactMolWt(mol),
            'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),

            # Atom counts
            'NumAtoms': mol.GetNumAtoms(),
            'NumHeavyAtoms': Lipinski.HeavyAtomCount(mol),
            'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),

            # Bonds
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
            'NumAromaticRings': Lipinski.NumAromaticRings(mol),
            'NumSaturatedRings': Lipinski.NumSaturatedRings(mol),

            # Hydrogen bonding
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),

            # Topological properties
            'TPSA': Descriptors.TPSA(mol),
            'LabuteASA': Descriptors.LabuteASA(mol),

            # Lipophilicity
            'MolLogP': Crippen.MolLogP(mol),
            'MolMR': Crippen.MolMR(mol),

            # Complexity
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),

            # Drug-likeness
            'QED': QED.qed(mol),

            # Additional descriptors
            'FractionCsp3': Lipinski.FractionCSP3(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'MinPartialCharge': Descriptors.MinPartialCharge(mol),
        }

        return descriptors
    except Exception as e:
        return None


def compute_fingerprints(smiles: str, fp_size: int = 2048) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprints

    Args:
        smiles: SMILES string
        fp_size: Fingerprint size

    Returns:
        Fingerprint array or None
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


def batch_compute_descriptors(smiles_list: List[str]) -> np.ndarray:
    """
    Compute descriptors for a batch of SMILES

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Array of descriptors (n_samples, n_features)
    """
    descriptors_list = []
    feature_names = None

    for smiles in smiles_list:
        desc = compute_rdkit_descriptors(smiles)

        if desc is None:
            # Use zeros for invalid molecules
            if feature_names is not None:
                desc = {name: 0.0 for name in feature_names}
            else:
                descriptors_list.append(None)
                continue

        if feature_names is None:
            feature_names = list(desc.keys())

        descriptors_list.append([desc[name] for name in feature_names])

    # Handle cases where some descriptors are None
    valid_descriptors = [d for d in descriptors_list if d is not None]
    if not valid_descriptors:
        return np.array([])

    # Replace None with mean values
    descriptors_array = []
    mean_values = np.mean(valid_descriptors, axis=0)

    for desc in descriptors_list:
        if desc is None:
            descriptors_array.append(mean_values)
        else:
            descriptors_array.append(desc)

    return np.array(descriptors_array), feature_names


def batch_compute_fingerprints(smiles_list: List[str], fp_size: int = 2048) -> np.ndarray:
    """
    Compute fingerprints for a batch of SMILES

    Args:
        smiles_list: List of SMILES strings
        fp_size: Fingerprint size

    Returns:
        Array of fingerprints (n_samples, fp_size)
    """
    fingerprints = []

    for smiles in smiles_list:
        fp = compute_fingerprints(smiles, fp_size)
        if fp is None:
            fp = np.zeros(fp_size)
        fingerprints.append(fp)

    return np.array(fingerprints)


class MolecularFeatureExtractor:
    """Extract molecular features from SMILES"""

    def __init__(self, use_descriptors: bool = True, use_fingerprints: bool = False, fp_size: int = 2048):
        """
        Initialize feature extractor

        Args:
            use_descriptors: Whether to compute RDKit descriptors
            use_fingerprints: Whether to compute Morgan fingerprints
            fp_size: Fingerprint size
        """
        self.use_descriptors = use_descriptors
        self.use_fingerprints = use_fingerprints
        self.fp_size = fp_size
        self.feature_names = None

    def fit(self, smiles_list: List[str]):
        """
        Fit feature extractor (determine feature names)

        Args:
            smiles_list: List of SMILES strings
        """
        if self.use_descriptors:
            # Get feature names from first valid molecule
            for smiles in smiles_list:
                desc = compute_rdkit_descriptors(smiles)
                if desc is not None:
                    self.feature_names = list(desc.keys())
                    break

        return self

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """
        Transform SMILES to features

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Feature array (n_samples, n_features)
        """
        features = []

        if self.use_descriptors:
            desc_features, _ = batch_compute_descriptors(smiles_list)
            features.append(desc_features)

        if self.use_fingerprints:
            fp_features = batch_compute_fingerprints(smiles_list, self.fp_size)
            features.append(fp_features)

        if not features:
            return np.array([])

        return np.concatenate(features, axis=1)

    def fit_transform(self, smiles_list: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(smiles_list)
        return self.transform(smiles_list)
