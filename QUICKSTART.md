# Quick Start Guide

## 1. Setup Environment

```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Explore the Data

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_eda.ipynb
```

Key insights from EDA:
- **7,973 training samples** with 5 target properties
- **Sparse targets**: Most samples don't have all 5 properties
- **SMILES with polymer notation**: Uses `*` for repeating units
- **Supplemental datasets**: ~10k additional training samples

## 3. Train the Model

### Single Training Run

```bash
cd src
python train.py --config ../configs/chemberta_baseline.yaml
```

### Cross-Validation (5-fold)

```bash
# Train on each fold
for fold in {0..4}; do
    python train.py --config ../configs/chemberta_baseline.yaml --fold $fold
done
```

### Monitor Training

```bash
# With TensorBoard
tensorboard --logdir experiments/chemberta_baseline/
```

## 4. Generate Predictions

```bash
cd src
python predict.py \
    --config ../configs/chemberta_baseline.yaml \
    --model ../experiments/chemberta_baseline/best_model_fold0.pt \
    --output ../submissions/
```

## 5. Submit to Kaggle

```bash
# Upload submission.csv to Kaggle
kaggle competitions submit \
    -c neurips-open-polymer-prediction-2025 \
    -f submissions/submission.csv \
    -m "ChemBERTa baseline"
```

## Model Architecture

```
Input: SMILES String (e.g., "*CC(*)c1ccccc1")
    ↓
ChemBERTa Tokenizer
    ↓
ChemBERTa Encoder (768-dim embeddings)
    ↓
[Optional: + RDKit Features]
    ↓
Shared Projection Layer (768-dim)
    ↓
    ├→ Tg Head → Prediction
    ├→ FFV Head → Prediction
    ├→ Tc Head → Prediction
    ├→ Density Head → Prediction
    └→ Rg Head → Prediction
```

## Configuration Options

Edit `configs/chemberta_baseline.yaml` to customize:

- **Model**: ChemBERTa variant, hidden dims, dropout
- **Training**: Epochs, batch size, learning rate
- **Features**: Enable/disable RDKit molecular descriptors
- **Data**: SMILES augmentation, supplemental datasets

## Tips for Improvement

1. **Ensemble models**: Train multiple folds and average predictions
2. **SMILES augmentation**: Enable in config for data augmentation
3. **Supplemental data**: Use domain adaptation for extra datasets
4. **Hyperparameter tuning**: Adjust learning rate, dropout, hidden dims
5. **Advanced models**: Try MegaMolBART or ensemble multiple transformers

## Troubleshooting

### RDKit Installation Issues

```bash
# Use conda instead
conda create -n polymer python=3.9
conda activate polymer
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

### CUDA Out of Memory

```bash
# Reduce batch size in config
# Change: batch_size: 32 → batch_size: 16
```

### Slow Training

```bash
# Enable mixed precision (already enabled in config)
# Verify: fp16: true

# Use smaller model
# Change: pretrained_model: "seyonec/ChemBERTa-zinc-base-v1"
# To: pretrained_model: "seyonec/ChemBERTa-zinc-base-v1" (smaller variant)
```

## Next Steps

1. Run EDA notebook to understand the data
2. Train baseline model with default config
3. Experiment with hyperparameters
4. Try SMILES augmentation
5. Ensemble multiple models
6. Submit to Kaggle!

Good luck!
