# NeurIPS Open Polymer Prediction 2025

Multi-task regression for predicting polymer properties from SMILES strings using pretrained molecular transformers.

## Competition Overview

Predict 5 polymer properties from SMILES molecular representations:
- **Tg**: Glass Transition Temperature
- **FFV**: Fractional Free Volume
- **Tc**: Crystallization Temperature
- **Density**: Material density
- **Rg**: Radius of Gyration

## Project Structure

```
neurips-polymer-prediction/
├── data/                          # Symlink to competition data
├── notebooks/                     # Jupyter notebooks for EDA and experiments
├── src/
│   ├── models/                   # Model architectures
│   ├── features/                 # Feature engineering & preprocessing
│   └── utils/                    # Helper functions
├── configs/                      # Configuration files
├── experiments/                  # Experiment logs and checkpoints
├── submissions/                  # Generated submission files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Link Competition Data

```bash
# Create symlink to competition data
ln -s ../neurips-open-polymer-prediction-2025 data/raw
```

### 3. Install RDKit (if pip fails)

RDKit can be tricky to install. If pip fails, use conda:

```bash
conda install -c conda-forge rdkit
```

## Model Approach

### Architecture
- **Base Model**: ChemBERTa-77M (pretrained on 77M SMILES from PubChem)
- **Task**: Multi-task regression with 5 property heads
- **Loss**: Weighted MSE (only on available targets)

### Training Strategy
1. SMILES tokenization with ChemBERTa tokenizer
2. Multi-task learning with shared encoder
3. 5-fold cross-validation
4. Data augmentation via SMILES enumeration
5. Ensemble with supplemental datasets

## Usage

### Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Training
```bash
python src/train.py --config configs/chemberta_baseline.yaml
```

### Inference
```bash
python src/predict.py --model experiments/best_model.pt --output submissions/
```

## Resources

- [ChemBERTa Paper](https://arxiv.org/abs/2010.09885)
- [ChemBERTa HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Competition Page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
