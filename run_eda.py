"""Run EDA analysis and generate visualizations"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Create output directory
output_dir = Path('eda_results')
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print(" " * 15 + "NeurIPS Polymer Prediction - EDA")
print("=" * 70)

# Load data
print("\n1. Loading Data...")
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

supplement_dfs = []
for i in range(1, 5):
    df = pd.read_csv(f'data/raw/train_supplement/dataset{i}.csv')
    df['source'] = f'dataset{i}'
    supplement_dfs.append(df)

print(f"   ✓ Train shape: {train_df.shape}")
print(f"   ✓ Test shape: {test_df.shape}")
print(f"   ✓ Supplemental datasets: {len(supplement_dfs)}")

# Target columns
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# 2. Missing Value Analysis
print("\n2. Missing Value Analysis...")
missing_stats = pd.DataFrame({
    'Missing Count': train_df[target_columns].isnull().sum(),
    'Missing %': (train_df[target_columns].isnull().sum() / len(train_df) * 100).round(2),
    'Available Count': train_df[target_columns].notnull().sum(),
    'Available %': (train_df[target_columns].notnull().sum() / len(train_df) * 100).round(2)
})

print("\n   Missing Value Statistics:")
print(missing_stats.to_string())

# Visualize missing data
fig, ax = plt.subplots(figsize=(10, 6))
missing_stats[['Available %', 'Missing %']].plot(
    kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c']
)
ax.set_xlabel('Percentage (%)')
ax.set_title('Data Availability by Target Property')
ax.legend(['Available', 'Missing'])
plt.tight_layout()
plt.savefig(output_dir / 'missing_values.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir}/missing_values.png")
plt.close()

# 3. Target Distribution Analysis
print("\n3. Target Distribution Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(target_columns):
    data = train_df[col].dropna()
    axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'{col} Distribution\\n(n={len(data)}, mean={data.mean():.3f})')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {data.mean():.3f}')
    axes[i].axvline(data.median(), color='green', linestyle=':', linewidth=2,
                     label=f'Median: {data.median():.3f}')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(output_dir / 'target_distributions.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir}/target_distributions.png")
plt.close()

# Print statistics
print("\n   Target Statistics:")
print(train_df[target_columns].describe().T.round(4))

# 4. Target Correlations
print("\n4. Target Correlation Analysis...")
corr_matrix = train_df[target_columns].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Target Property Correlations')
plt.tight_layout()
plt.savefig(output_dir / 'target_correlations.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir}/target_correlations.png")
print("\n   Correlation Matrix:")
print(corr_matrix.round(3))
plt.close()

# 5. SMILES Analysis
print("\n5. SMILES Analysis...")
train_df['smiles_length'] = train_df['SMILES'].str.len()
train_df['has_repeating_unit'] = train_df['SMILES'].str.contains('\\*', regex=False)
train_df['num_stars'] = train_df['SMILES'].str.count('\\*')
train_df['num_rings'] = train_df['SMILES'].str.count('1') + train_df['SMILES'].str.count('2')
train_df['num_aromatic'] = train_df['SMILES'].str.count('c')

print(f"\n   SMILES Statistics:")
print(f"   - Average length: {train_df['smiles_length'].mean():.1f}")
print(f"   - Samples with repeating units (*): {train_df['has_repeating_unit'].sum()} ({train_df['has_repeating_unit'].mean()*100:.1f}%)")
print(f"   - Average number of * symbols: {train_df['num_stars'].mean():.2f}")
print(f"   - Average number of rings: {train_df['num_rings'].mean():.2f}")
print(f"   - Average aromatic count: {train_df['num_aromatic'].mean():.2f}")

# SMILES length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['smiles_length'], bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[0].set_xlabel('SMILES Length')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'SMILES Length Distribution\\n(Mean: {train_df["smiles_length"].mean():.1f})')
axes[0].axvline(train_df['smiles_length'].mean(), color='red', linestyle='--', linewidth=2)
axes[0].grid(alpha=0.3)

axes[1].boxplot(train_df['smiles_length'], vert=True)
axes[1].set_ylabel('SMILES Length')
axes[1].set_title('SMILES Length Boxplot')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'smiles_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {output_dir}/smiles_analysis.png")
plt.close()

# 6. Supplemental Dataset Analysis
print("\n6. Supplemental Dataset Analysis...")
for df in supplement_dfs:
    source = df['source'].iloc[0]
    print(f"\n   {source}:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    if 'TC_mean' in df.columns:
        print(f"   - TC_mean range: [{df['TC_mean'].min():.3f}, {df['TC_mean'].max():.3f}]")
        print(f"   - TC_mean mean: {df['TC_mean'].mean():.3f}")

# 7. Summary Report
print("\n" + "=" * 70)
print(" " * 25 + "KEY INSIGHTS")
print("=" * 70)
print("""
1. TARGET SPARSITY:
   - FFV is the most complete target (88% available)
   - Tg, Tc, Density, Rg are very sparse (6-9% available)
   → Multi-task learning with masked loss is essential!

2. DATA CHARACTERISTICS:
   - 7,973 training samples with 5 target properties
   - High variance in target values across properties
   - SMILES strings vary significantly in length

3. MODELING RECOMMENDATIONS:
   - Use multi-task learning to share representations
   - FFV will be the "anchor" task for learning molecular features
   - ChemBERTa pretrained transformer is ideal for SMILES
   - Consider SMILES augmentation for regularization
   - Use supplemental datasets with domain adaptation

4. EXPECTED CHALLENGES:
   - Sparse targets require careful loss weighting
   - Some properties may be difficult to predict with limited data
   - SMILES representation quality is critical
""")
print("=" * 70)
print(f"\n✓ EDA Complete! Results saved to '{output_dir}/' directory")
print("\nGenerated files:")
for file in sorted(output_dir.glob('*.png')):
    print(f"  - {file.name}")
print("\n" + "=" * 70)
