#!/bin/bash
# Setup script for NeurIPS Polymer Prediction project

echo "Setting up NeurIPS Polymer Prediction project..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Try installing rdkit via pip first
echo "Installing RDKit..."
pip install rdkit || echo "Warning: RDKit installation via pip failed. You may need to use conda: conda install -c conda-forge rdkit"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start exploring the data, run:"
echo "  jupyter notebook notebooks/01_eda.ipynb"
echo ""
echo "To train the model, run:"
echo "  python src/train.py --config configs/chemberta_baseline.yaml"
