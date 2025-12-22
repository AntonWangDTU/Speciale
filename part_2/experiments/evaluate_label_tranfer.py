#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Evaluate label transfer performance across all grid search configurations
INCLUDING preprocessed hvg gene expression data
"""
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from itertools import product
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_TYPE = 'GSE114412_cluster'    
DATA_PATH = '../data'
RESULTS_PATH = '../results'

# If DATA argument is provided OVERRIDE
if len(sys.argv) >= 2:
    DATA_TYPE = sys.argv[1]
print(f"Using data: {DATA_TYPE}")

# Grid search parameters
LATENT_DIMS = [10, 20, 30, 40, 50]
DENOISE_STEPS = [50, 100, 150, 200, 250]

# Classifier to use
CLASSIFIER = 'logistic_regression'  # or 'random_forest'

# Create output directory
os.makedirs(os.path.join(RESULTS_PATH, 'label_transfer'), exist_ok=True)

print("="*60)
print(f"LABEL TRANSFER EVALUATION - {CLASSIFIER.upper()}")
print("="*60)

# ============================================================
# Load all three splits
# ============================================================
print(f"\nLoading data splits for {DATA_TYPE}...")
adata_train = sc.read_h5ad(os.path.join(DATA_PATH, f'{DATA_TYPE}_grid_search_all_results_train.h5ad'))
adata_val = sc.read_h5ad(os.path.join(DATA_PATH, f'{DATA_TYPE}_grid_search_all_results_val.h5ad'))
adata_test = sc.read_h5ad(os.path.join(DATA_PATH, f'{DATA_TYPE}_grid_search_all_results_test.h5ad'))

print(f"✓ Loaded:")
print(f"  Train: {adata_train.n_obs} cells")
print(f"  Val:   {adata_val.n_obs} cells")
print(f"  Test:  {adata_test.n_obs} cells")

print(f"\n✓ Available representations: {len(adata_train.obsm)}")

# ============================================================
# Preprocess hvg data for all splits (WITHOUT DATA LEAKAGE)
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING hvg DATA FOR LABEL TRANSFER")
print("="*60)

# Create copies for preprocessing
print("Creating copies for preprocessing...")
adata_train_hvg = adata_train.copy()
adata_val_hvg = adata_val.copy()
adata_test_hvg = adata_test.copy()

# Step 1: Normalize each split independently (per-cell operation, no leakage)
print("  1. Normalizing to median total counts (target_sum=1e4)...")
sc.pp.normalize_total(adata_train_hvg, target_sum=1e4)
sc.pp.normalize_total(adata_val_hvg, target_sum=1e4)
sc.pp.normalize_total(adata_test_hvg, target_sum=1e4)

# Step 2: Log-transform each split independently (per-cell operation, no leakage)
print("  2. Log-transforming (log1p)...")
sc.pp.log1p(adata_train_hvg)
sc.pp.log1p(adata_val_hvg)
sc.pp.log1p(adata_test_hvg)

# Step 3: Select HVGs from TRAINING set only, apply to all splits
print("  3. Selecting highly variable genes from TRAINING set only...")
sc.pp.highly_variable_genes(
    adata_train_hvg,
    n_top_genes=2000,
    subset=False  # Don't subset yet, just mark
)

# Get HVG names from training set
hvg_genes = adata_train_hvg.var_names[adata_train_hvg.var['highly_variable']].tolist()
print(f"     Selected {len(hvg_genes)} HVGs from training set")

# Subset all splits to the SAME genes (from training set)
print("  4. Applying HVG selection to all splits...")
adata_train_hvg = adata_train_hvg[:, hvg_genes].copy()
adata_val_hvg = adata_val_hvg[:, hvg_genes].copy()
adata_test_hvg = adata_test_hvg[:, hvg_genes].copy()

print(f"✓ Preprocessed shape: {adata_train_hvg.shape}")
print(f"  (cells × HVGs: {adata_train_hvg.n_obs} × {adata_train_hvg.n_vars})")

# Convert to dense arrays for classifier
X_train_hvg = adata_train_hvg.X.toarray() if hasattr(adata_train_hvg.X, 'toarray') else adata_train_hvg.X
X_val_hvg = adata_val_hvg.X.toarray() if hasattr(adata_val_hvg.X, 'toarray') else adata_val_hvg.X
X_test_hvg = adata_test_hvg.X.toarray() if hasattr(adata_test_hvg.X, 'toarray') else adata_test_hvg.X

print(f"✓ Converted to dense arrays")

# ============================================================
# Build list of all representations to test
# ============================================================
print("\n" + "="*60)
print("BUILDING REPRESENTATION LIST")
print("="*60)

representations = []

# 0. hvg preprocessed data (NEW!)
representations.append({
    'name': 'X_hvg_processed',
    'type': 'hvg',
    'latent_dim': None,
    'denoise_steps': None,
    'data_train': X_train_hvg,
    'data_val': X_val_hvg,
    'data_test': X_test_hvg
})

# 1. PCA for each latent dimension
for latent_dim in LATENT_DIMS:
    pca_key = f'X_pca_latent{latent_dim}'
    if pca_key in adata_train.obsm:
        representations.append({
            'name': pca_key,
            'type': 'pca',
            'latent_dim': latent_dim,
            'denoise_steps': None,
            'data_train': adata_train.obsm[pca_key],
            'data_val': adata_val.obsm[pca_key],
            'data_test': adata_test.obsm[pca_key]
        })

# 2. VAE for each latent dimension
for latent_dim in LATENT_DIMS:
    vae_key = f'X_VAE_latent{latent_dim}'
    if vae_key in adata_train.obsm:
        representations.append({
            'name': vae_key,
            'type': 'vae',
            'latent_dim': latent_dim,
            'denoise_steps': None,
            'data_train': adata_train.obsm[vae_key],
            'data_val': adata_val.obsm[vae_key],
            'data_test': adata_test.obsm[vae_key]
        })

# 3. Denoised for all combinations
for latent_dim, denoise_steps in product(LATENT_DIMS, DENOISE_STEPS):
    denoised_key = f'X_VAE_denoised_latent{latent_dim}_steps{denoise_steps}'
    if denoised_key in adata_train.obsm:
        representations.append({
            'name': denoised_key,
            'type': 'denoised',
            'latent_dim': latent_dim,
            'denoise_steps': denoise_steps,
            'data_train': adata_train.obsm[denoised_key],
            'data_val': adata_val.obsm[denoised_key],
            'data_test': adata_test.obsm[denoised_key]
        })

print(f"\n✓ Found {len(representations)} representations to evaluate")
print(f"  hvg: {sum(1 for r in representations if r['type'] == 'hvg')}")
print(f"  PCA: {sum(1 for r in representations if r['type'] == 'pca')}")
print(f"  VAE: {sum(1 for r in representations if r['type'] == 'vae')}")
print(f"  Denoised: {sum(1 for r in representations if r['type'] == 'denoised')}")

# ============================================================
# Helper function to get classifier
# ============================================================
def get_classifier(classifier_type='logistic_regression'):
    """Get classifier instance"""
    if classifier_type == 'logistic_regression':
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            solver='lbfgs',
            multi_class='multinomial'
        )
    elif classifier_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=20
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

# ============================================================
# Evaluate each representation
# ============================================================
results = []

# Get labels
y_train = adata_train.obs['cell_type'].values
y_val = adata_val.obs['cell_type'].values
y_test = adata_test.obs['cell_type'].values

print(f"\n{'='*60}")
print(f"STARTING EVALUATION WITH {CLASSIFIER.upper()}")
print(f"{'='*60}\n")

for idx, rep_info in enumerate(tqdm(representations, desc="Evaluating representations")):
    rep = rep_info['name']
    rep_type = rep_info['type']
    latent_dim = rep_info['latent_dim']
    denoise_steps = rep_info['denoise_steps']
    
    # Get pre-loaded data
    X_train = rep_info['data_train']
    X_val = rep_info['data_val']
    X_test = rep_info['data_test']
    
    print(f"\n[{idx+1}/{len(representations)}] Evaluating: {rep}")
    print(f"  Shape: {X_train.shape}")
    
    try:
        # Check for NaN or inf
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            nan_count = np.isnan(X_train).sum()
            inf_count = np.isinf(X_train).sum()
            print(f"  ⚠️  Skipping: NaN={nan_count}, Inf={inf_count} in training set")
            continue
        
        print(f"  Training {CLASSIFIER}...")
        
        # Train classifier
        clf = get_classifier(CLASSIFIER)
        clf.fit(X_train, y_train)
        
        # Predict on validation set
        print(f"  Predicting on validation set...")
        y_val_pred = clf.predict(X_val)
        
        # Predict on test set
        print(f"  Predicting on test set...")
        y_test_pred = clf.predict(X_test)
        
        # Compute metrics
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        
        val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
        
        # Store results
        result = {
            'representation': rep,
            'type': rep_type,
            'latent_dim': latent_dim if latent_dim is not None else 'N/A',
            'denoise_steps': denoise_steps if denoise_steps is not None else 'N/A',
            'classifier': CLASSIFIER,
            'n_features': X_train.shape[1],
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'val_f1_weighted': val_f1_weighted,
            'test_f1_weighted': test_f1_weighted,
            'val_f1_macro': val_f1_macro,
            'test_f1_macro': test_f1_macro,
            'val_balanced_accuracy': val_balanced_acc,
            'test_balanced_accuracy': test_balanced_acc
        }
        
        results.append(result)
        
        print(f"  Val  → Acc: {val_acc:.4f}, F1(w): {val_f1_weighted:.4f}, F1(m): {val_f1_macro:.4f}")
        print(f"  Test → Acc: {test_acc:.4f}, F1(w): {test_f1_weighted:.4f}, F1(m): {test_f1_macro:.4f}")
        
        # Save intermediate results
        df_results = pd.DataFrame(results)
        df_results.to_csv(
            os.path.join(RESULTS_PATH, 'label_transfer', 
                        f'label_transfer_results_{DATA_TYPE}_{CLASSIFIER}.csv'),
            index=False
        )
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("LABEL TRANSFER EVALUATION COMPLETE")
print("="*60)

df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("\n❌ No results generated. Check errors above.")
    exit(1)

print(f"\n✓ Successfully evaluated {len(df_results)} configurations")

# Save final results
final_csv = os.path.join(RESULTS_PATH, 'label_transfer', 
                         f'label_transfer_results_{DATA_TYPE}_{CLASSIFIER}_final.csv')
df_results.to_csv(final_csv, index=False)
print(f"✓ Saved results to: {final_csv}")

# ============================================================
# Summary by type
# ============================================================
print("\n" + "-"*60)
print("SUMMARY BY REPRESENTATION TYPE")
print("-"*60)

for rep_type in ['hvg', 'pca', 'vae', 'denoised']:
    subset = df_results[df_results['type'] == rep_type]
    if len(subset) > 0:
        print(f"\n{rep_type.upper()} (n={len(subset)}):")
        print(f"  Test Accuracy:      {subset['test_accuracy'].mean():.4f} ± {subset['test_accuracy'].std():.4f}")
        print(f"                      [min: {subset['test_accuracy'].min():.4f}, max: {subset['test_accuracy'].max():.4f}]")
        print(f"  Test F1 (weighted): {subset['test_f1_weighted'].mean():.4f} ± {subset['test_f1_weighted'].std():.4f}")
        print(f"                      [min: {subset['test_f1_weighted'].min():.4f}, max: {subset['test_f1_weighted'].max():.4f}]")
        print(f"  Test F1 (macro):    {subset['test_f1_macro'].mean():.4f} ± {subset['test_f1_macro'].std():.4f}")
        print(f"                      [min: {subset['test_f1_macro'].min():.4f}, max: {subset['test_f1_macro'].max():.4f}]")

# ============================================================
# Best overall configuration
# ============================================================
print("\n" + "-"*60)
print("TOP 10 CONFIGURATIONS (by test F1 weighted)")
print("-"*60)
top10 = df_results.nlargest(10, 'test_f1_weighted')[
    ['representation', 'type', 'latent_dim', 'denoise_steps', 
     'n_features', 'test_accuracy', 'test_f1_weighted', 'test_f1_macro']
]
print(top10.to_string(index=False))

# ============================================================
# Best per type
# ============================================================
print("\n" + "-"*60)
print("BEST CONFIGURATION PER TYPE")
print("-"*60)

for rep_type in ['hvg', 'pca', 'vae', 'denoised']:
    subset = df_results[df_results['type'] == rep_type]
    if len(subset) > 0:
        best_idx = subset['test_f1_weighted'].idxmax()
        best = subset.loc[best_idx]
        print(f"\n{rep_type.upper()}:")
        print(f"  Representation: {best['representation']}")
        print(f"  Latent Dim: {best['latent_dim']}")
        print(f"  Denoise Steps: {best['denoise_steps']}")
        print(f"  N Features: {best['n_features']}")
        print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        print(f"  Test F1 (weighted): {best['test_f1_weighted']:.4f}")
        print(f"  Test F1 (macro): {best['test_f1_macro']:.4f}")
        print(f"  Test Balanced Acc: {best['test_balanced_accuracy']:.4f}")

# Overall best
best_idx = df_results['test_f1_weighted'].idxmax()
best_overall = df_results.loc[best_idx]

print("\n" + "="*60)
print("BEST OVERALL CONFIGURATION")
print("="*60)
print(f"Representation: {best_overall['representation']}")
print(f"Type: {best_overall['type']}")
print(f"Latent Dim: {best_overall['latent_dim']}")
print(f"Denoise Steps: {best_overall['denoise_steps']}")
print(f"N Features: {best_overall['n_features']}")
print(f"Test Accuracy: {best_overall['test_accuracy']:.4f}")
print(f"Test F1 (weighted): {best_overall['test_f1_weighted']:.4f}")
print(f"Test F1 (macro): {best_overall['test_f1_macro']:.4f}")
print(f"Test Balanced Acc: {best_overall['test_balanced_accuracy']:.4f}")
print("="*60)

# ============================================================
# Compute improvement over hvg baseline
# ============================================================
print("\n" + "-"*60)
print("IMPROVEMENT OVER hvg BASELINE")
print("-"*60)

hvg_result = df_results[df_results['type'] == 'hvg']
if len(hvg_result) > 0:
    hvg_f1 = hvg_result['test_f1_weighted'].values[0]
    hvg_acc = hvg_result['test_accuracy'].values[0]
    
    print(f"\nhvg baseline:")
    print(f"  Test Accuracy: {hvg_acc:.4f}")
    print(f"  Test F1 (weighted): {hvg_f1:.4f}")
    
    # Show improvement for each type
    for rep_type in ['pca', 'vae', 'denoised']:
        subset = df_results[df_results['type'] == rep_type]
        if len(subset) > 0:
            best_idx = subset['test_f1_weighted'].idxmax()
            best = subset.loc[best_idx]
            
            improvement_f1 = best['test_f1_weighted'] - hvg_f1
            improvement_acc = best['test_accuracy'] - hvg_acc
            improvement_f1_pct = (improvement_f1 / hvg_f1) * 100
            improvement_acc_pct = (improvement_acc / hvg_acc) * 100
            
            print(f"\nBest {rep_type.upper()}:")
            print(f"  F1 improvement: +{improvement_f1:.4f} ({improvement_f1_pct:+.2f}%)")
            print(f"  Accuracy improvement: +{improvement_acc:.4f} ({improvement_acc_pct:+.2f}%)")

# ============================================================
# Compare with cLISI results
# ============================================================
print("\n" + "="*60)
print("COMPARING WITH cLISI RESULTS")
print("="*60)

clisi_csv = os.path.join(RESULTS_PATH, f'grid_search_results_{DATA_TYPE}_final.csv')
if os.path.exists(clisi_csv):
    df_clisi = pd.read_csv(clisi_csv)
    
    # Filter for denoised results only
    df_denoised = df_results[df_results['type'] == 'denoised'].copy()
    
    # Ensure correct types
    df_denoised['latent_dim_int'] = pd.to_numeric(df_denoised['latent_dim'], errors='coerce').astype('Int64')
    df_denoised['denoise_steps_int'] = pd.to_numeric(df_denoised['denoise_steps'], errors='coerce').astype('Int64')
    
    # Merge
    df_merged = df_denoised.merge(
        df_clisi[['latent_dim', 'denoise_steps', 'clisi_pca', 'clisi_vae', 'clisi_denoised', 'clisi_hvg']],
        left_on=['latent_dim_int', 'denoise_steps_int'],
        right_on=['latent_dim', 'denoise_steps'],
        how='inner'
    )
    
    if len(df_merged) > 0:
        print(f"\n✓ Successfully merged {len(df_merged)} configurations")
        
        # Compute correlations
        from scipy.stats import pearsonr
        
        corr_f1_weighted, p_f1_weighted = pearsonr(df_merged['test_f1_weighted'], 
                                                    df_merged['clisi_denoised'])
        corr_f1_macro, p_f1_macro = pearsonr(df_merged['test_f1_macro'], 
                                              df_merged['clisi_denoised'])
        corr_acc, p_acc = pearsonr(df_merged['test_accuracy'], 
                                    df_merged['clisi_denoised'])
        
        print(f"\nCorrelation between cLISI (Denoised) and Label Transfer:")
        print(f"  cLISI vs Test F1 (weighted): r={corr_f1_weighted:.4f}, p={p_f1_weighted:.2e}")
        print(f"  cLISI vs Test F1 (macro):    r={corr_f1_macro:.4f}, p={p_f1_macro:.2e}")
        print(f"  cLISI vs Test Accuracy:      r={corr_acc:.4f}, p={p_acc:.2e}")
        
        # Statistical significance
        print(f"\nSignificance:")
        for name, p_val in [('F1 weighted', p_f1_weighted), 
                           ('F1 macro', p_f1_macro), 
                           ('Accuracy', p_acc)]:
            if p_val < 0.001:
                sig = "*** (p < 0.001)"
            elif p_val < 0.01:
                sig = "** (p < 0.01)"
            elif p_val < 0.05:
                sig = "* (p < 0.05)"
            else:
                sig = "ns (p >= 0.05)"
            print(f"  {name}: {sig}")
        
        # Save merged results
        df_merged.to_csv(
            os.path.join(RESULTS_PATH, 'label_transfer', 
                        f'combined_clisi_labeltransfer_{DATA_TYPE}_{CLASSIFIER}.csv'),
            index=False
        )
        print(f"\n✓ Saved merged results")
        
        # Combined score analysis
        df_merged['combined_score_f1'] = (
            df_merged['clisi_denoised'] + df_merged['test_f1_weighted']
        ) / 2
        
        df_merged['combined_score_acc'] = (
            df_merged['clisi_denoised'] + df_merged['test_accuracy']
        ) / 2
        
        print(f"\n" + "-"*60)
        print("TOP 5 BY COMBINED SCORE (cLISI + Label Transfer F1)")
        print("-"*60)
        top_combined = df_merged.nlargest(5, 'combined_score_f1')[
            ['latent_dim_int', 'denoise_steps_int', 'clisi_denoised', 
             'test_f1_weighted', 'combined_score_f1']
        ]
        top_combined.columns = ['Latent Dim', 'Denoise Steps', 'cLISI', 'Test F1', 'Combined']
        print(top_combined.to_string(index=False))
        
        print(f"\n" + "-"*60)
        print("TOP 5 BY COMBINED SCORE (cLISI + Label Transfer Accuracy)")
        print("-"*60)
        top_combined_acc = df_merged.nlargest(5, 'combined_score_acc')[
            ['latent_dim_int', 'denoise_steps_int', 'clisi_denoised', 
             'test_accuracy', 'combined_score_acc']
        ]
        top_combined_acc.columns = ['Latent Dim', 'Denoise Steps', 'cLISI', 'Test Acc', 'Combined']
        print(top_combined_acc.to_string(index=False))
        
        # Check if top configs are consistent
        print(f"\n" + "-"*60)
        print("CONSISTENCY CHECK")
        print("-"*60)
        
        top5_by_clisi = set(df_merged.nlargest(5, 'clisi_denoised').apply(
            lambda x: (int(x['latent_dim_int']), int(x['denoise_steps_int'])), axis=1
        ))
        top5_by_f1 = set(df_merged.nlargest(5, 'test_f1_weighted').apply(
            lambda x: (int(x['latent_dim_int']), int(x['denoise_steps_int'])), axis=1
        ))
        
        overlap = top5_by_clisi.intersection(top5_by_f1)
        
        print(f"  Top 5 by cLISI: {sorted(top5_by_clisi)}")
        print(f"  Top 5 by Label Transfer F1: {sorted(top5_by_f1)}")
        print(f"  Overlap: {len(overlap)}/5 configurations")
        
        if overlap:
            print(f"  Consistent configs: {sorted(overlap)}")
        
else:
    print(f"⚠️  Could not find cLISI results at: {clisi_csv}")
    print("    Run the main grid search first to generate cLISI results")

print("\n✓ Label transfer evaluation complete!")
print(f"\n📁 Results saved to:")
print(f"  CSV: {final_csv}")
print(f"  Plots: Run plot_label_transfer.py to generate visualizations")