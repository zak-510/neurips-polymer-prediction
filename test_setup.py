"""Quick test to verify setup and data loading"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_data_loading():
    """Test that data loads correctly"""
    print("=" * 60)
    print("Testing Data Loading...")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')

    print(f"\n✓ Train data shape: {train_df.shape}")
    print(f"✓ Test data shape: {test_df.shape}")

    # Check columns
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print(f"\n✓ Target columns: {target_cols}")

    # Check missing values
    print("\nMissing value counts:")
    for col in target_cols:
        missing = train_df[col].isna().sum()
        available = train_df[col].notna().sum()
        print(f"  {col:10s}: {available:4d} available, {missing:4d} missing ({missing/len(train_df)*100:.1f}%)")

    return train_df, test_df


def test_smiles_processing():
    """Test SMILES processing with RDKit"""
    print("\n" + "=" * 60)
    print("Testing SMILES Processing...")
    print("=" * 60)

    from rdkit import Chem

    # Test with a sample SMILES
    sample_smiles = "*CC(*)c1ccccc1"
    print(f"\nSample SMILES: {sample_smiles}")

    mol = Chem.MolFromSmiles(sample_smiles)
    if mol:
        print("✓ Successfully parsed SMILES with RDKit")
        print(f"  Atoms: {mol.GetNumAtoms()}")
        print(f"  Bonds: {mol.GetNumBonds()}")
    else:
        print("✗ Failed to parse SMILES")
        return False

    return True


def test_model_loading():
    """Test that we can load ChemBERTa tokenizer"""
    print("\n" + "=" * 60)
    print("Testing Model Components...")
    print("=" * 60)

    from transformers import AutoTokenizer

    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    print(f"\nLoading tokenizer: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Successfully loaded ChemBERTa tokenizer")

        # Test tokenization
        sample_smiles = "*CC(*)c1ccccc1"
        encoding = tokenizer(sample_smiles, return_tensors='pt')
        print(f"✓ Tokenized SMILES into {encoding['input_ids'].shape[1]} tokens")

        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False


def test_molecular_features():
    """Test molecular feature extraction"""
    print("\n" + "=" * 60)
    print("Testing Molecular Features...")
    print("=" * 60)

    from features.smiles_preprocessing import get_polymer_features, validate_smiles

    sample_smiles = "*CC(*)c1ccccc1"
    print(f"\nSample SMILES: {sample_smiles}")

    # Validate
    is_valid = validate_smiles(sample_smiles)
    print(f"✓ SMILES validation: {is_valid}")

    # Extract features
    features = get_polymer_features(sample_smiles)
    print("\nPolymer features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    return True


def main():
    """Run all tests"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     NeurIPS Polymer Prediction - Setup Verification      ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    all_passed = True

    try:
        # Test 1: Data loading
        train_df, test_df = test_data_loading()

        # Test 2: SMILES processing
        if not test_smiles_processing():
            all_passed = False

        # Test 3: Model loading
        if not test_model_loading():
            all_passed = False

        # Test 4: Feature extraction
        if not test_molecular_features():
            all_passed = False

        # Final summary
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
            print("\nYou're ready to start training!")
            print("\nNext steps:")
            print("  1. jupyter notebook notebooks/01_eda.ipynb")
            print("  2. cd src && python train.py --config ../configs/chemberta_baseline.yaml")
        else:
            print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
