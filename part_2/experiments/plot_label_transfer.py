#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Visualize label transfer results from scikit-learn classifiers
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import pearsonr

# Configuration
DATA_TYPE = 'allCT_1.0k'
RESULTS_PATH = '../results'
CLASSIFIER = 'logistic_regression'  # Match your evaluation script

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Load results
csv_file = os.path.join(RESULTS_PATH, 'label_transfer', 
                       f'label_transfer_results_{DATA_TYPE}_{CLASSIFIER}_final.csv')
print(f"Loading results from: {csv_file}")

if not os.path.exists(csv_file):
    print(f"❌ File not found: {csv_file}")
    print("Run evaluate_label_transfer.py first!")
    exit(1)

df = pd.read_csv(csv_file)

df.iloc[0,1] = 'hvg'
print(f"✓ Loaded {len(df)} experiments")
print(f"  Columns: {df.columns.tolist()}")

# ============================================================
# PLOT 0: Baseline Comparison (Raw vs Best of Each Type)
# ============================================================
print("Generating baseline comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get best configuration for each type
baseline_data = []
for rep_type in ['hvg', 'raw', 'pca', 'vae', 'denoised']:
    subset = df[df['type'] == rep_type]
    if len(subset) > 0:
        best_idx = subset['test_f1_weighted'].idxmax()
        best = subset.loc[best_idx]
        label = rep_type.upper()
        if rep_type == 'denoised':
            label = f"Denoised\nL{int(best['latent_dim'])}_S{int(best['denoise_steps'])}"
        elif rep_type in ['pca', 'vae']:
            label = f"{rep_type.upper()}\nL{int(best['latent_dim'])}"
        elif rep_type == 'hvg':
            label = 'HVG'
        elif rep_type == 'raw':
            label = 'Raw'
        
        baseline_data.append({
            'method': label,
            'type': rep_type,
            'test_accuracy': best['test_accuracy'],
            'test_f1_weighted': best['test_f1_weighted'],
            'n_features': best['n_features']
        })

df_baseline = pd.DataFrame(baseline_data)

# Plot accuracy
colors_baseline = ['#9b59b6', '#e74c3c', '#3498db', '#f39c12', '#2ecc71']  # Added purple for HVG
bars1 = axes[0].bar(range(len(df_baseline)), df_baseline['test_accuracy'],
                    color=colors_baseline[:len(df_baseline)], alpha=0.8)
axes[0].set_xticks(range(len(df_baseline)))
axes[0].set_xticklabels(df_baseline['method'], fontsize=10)
axes[0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Best Test Accuracy by Representation Type', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels and feature count
for i, (bar, row) in enumerate(zip(bars1, df_baseline.iterrows())):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{row[1]["test_accuracy"]:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].text(bar.get_x() + bar.get_width()/2., height - 0.015,
                f'{int(row[1]["n_features"])} feat',
                ha='center', va='top', fontsize=7, style='italic', color='white')

# Plot F1
bars2 = axes[1].bar(range(len(df_baseline)), df_baseline['test_f1_weighted'],
                    color=colors_baseline[:len(df_baseline)], alpha=0.8)
axes[1].set_xticks(range(len(df_baseline)))
axes[1].set_xticklabels(df_baseline['method'], fontsize=10)
axes[1].set_ylabel('Test F1 (Weighted)', fontsize=12, fontweight='bold')
axes[1].set_title('Best Test F1 by Representation Type', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels and feature count
for i, (bar, row) in enumerate(zip(bars2, df_baseline.iterrows())):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{row[1]["test_f1_weighted"]:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1].text(bar.get_x() + bar.get_width()/2., height - 0.015,
                f'{int(row[1]["n_features"])} feat',
                ha='center', va='top', fontsize=7, style='italic', color='white')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer',
                        f'0_baseline_comparison_{DATA_TYPE}_{CLASSIFIER}.png'),
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: baseline_comparison_{DATA_TYPE}_{CLASSIFIER}.png")
plt.close()

# ============================================================
# PLOT 1: Line plots - Denoise steps on X-axis, color by latent_dim
# ============================================================
print("\nGenerating line plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Filter for denoised results only
df_denoised = df[df['type'] == 'denoised'].copy()

if len(df_denoised) > 0:
    # Ensure correct types
    df_denoised['latent_dim'] = pd.to_numeric(df_denoised['latent_dim'], errors='coerce').astype(int)
    df_denoised['denoise_steps'] = pd.to_numeric(df_denoised['denoise_steps'], errors='coerce').astype(int)
    
    # Define metrics to plot
    metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('test_f1_weighted', 'Test F1 (Weighted)'),
        ('test_f1_macro', 'Test F1 (Macro)'),
        ('test_balanced_accuracy', 'Test Balanced Accuracy')
    ]
    
    # Create color palette
    latent_dims = sorted(df_denoised['latent_dim'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(latent_dims)))
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in df_denoised.columns:
            ax.text(0.5, 0.5, f'{metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Plot line for each latent dimension
        for latent_dim, color in zip(latent_dims, colors):
            subset = df_denoised[df_denoised['latent_dim'] == latent_dim].sort_values('denoise_steps')
            ax.plot(subset['denoise_steps'], subset[metric], 
                   marker='o', linewidth=2.5, markersize=9,
                   label=f'Latent {latent_dim}', color=color)
        
        ax.set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title='Latent Dimension', fontsize=10, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([df_denoised[metric].min() - 0.01, df_denoised[metric].max() + 0.01])
        
        # Highlight best point
        best_idx = df_denoised[metric].idxmax()
        best_row = df_denoised.loc[best_idx]
        ax.scatter(best_row['denoise_steps'], best_row[metric], 
                  s=300, color='red', marker='*', zorder=5,
                  edgecolors='black', linewidths=2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                        f'1_label_transfer_lineplot_{DATA_TYPE}_{CLASSIFIER}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: label_transfer_lineplot_{DATA_TYPE}_{CLASSIFIER}.png")
plt.close()

# ============================================================
# PLOT 2: Heatmaps for denoised (4 metrics)
# ============================================================
print("Generating heatmaps...")

if len(df_denoised) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    heatmap_metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('test_f1_weighted', 'Test F1 (Weighted)'),
        ('test_f1_macro', 'Test F1 (Macro)'),
        ('test_balanced_accuracy', 'Test Balanced Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(heatmap_metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in df_denoised.columns:
            continue
        
        # Create pivot table
        pivot = df_denoised.pivot(index='latent_dim', 
                                 columns='denoise_steps', 
                                 values=metric)
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Score'},
                   linewidths=0.5, linecolor='white', ax=ax,
                   vmin=pivot.min().min(), vmax=pivot.max().max())
        
        ax.set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Highlight best configuration
        if idx == 1:  # Add red box on F1 weighted plot
            best_idx = df_denoised[metric].idxmax()
            best_config = df_denoised.loc[best_idx]
            best_latent = int(best_config['latent_dim'])
            best_steps = int(best_config['denoise_steps'])
            
            row_idx = list(pivot.index).index(best_latent)
            col_idx = list(pivot.columns).index(best_steps)
            
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, 
                                fill=False, edgecolor='red', linewidth=4)
            ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                            f'2_label_transfer_heatmaps_{DATA_TYPE}_{CLASSIFIER}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: label_transfer_heatmaps_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# PLOT 3: Comparison across representation types (PCA, VAE, Denoised)
# ============================================================
print("Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get baselines (PCA and VAE)
df_pca = df[df['type'] == 'pca'].copy()
df_vae = df[df['type'] == 'vae'].copy()

# Metrics to compare
compare_metrics = [
    ('test_accuracy', 'Test Accuracy'),
    ('test_f1_weighted', 'Test F1 (Weighted)'),
    ('test_f1_macro', 'Test F1 (Macro)'),
    ('test_balanced_accuracy', 'Test Balanced Accuracy')
]

for idx, (metric, title) in enumerate(compare_metrics):
    ax = axes[idx // 2, idx % 2]
    
    if metric not in df.columns:
        continue
    
    # Get best denoised for each latent dim
    best_denoised_per_latent = []
    for latent_dim in sorted(df_denoised['latent_dim'].unique()):
        subset = df_denoised[df_denoised['latent_dim'] == latent_dim]
        best_idx = subset[metric].idxmax()
        best = subset.loc[best_idx]
        best_denoised_per_latent.append({
            'latent_dim': latent_dim,
            'score': best[metric],
            'denoise_steps': best['denoise_steps']
        })
    
    df_best_denoised = pd.DataFrame(best_denoised_per_latent)
    
    # Plot
    x = np.arange(len(latent_dims))
    width = 0.25
    
    # PCA baseline
    if len(df_pca) > 0:
        pca_scores = [df_pca[df_pca['latent_dim'] == ld][metric].values[0] 
                     if len(df_pca[df_pca['latent_dim'] == ld]) > 0 else 0 
                     for ld in latent_dims]
        ax.bar(x - width, pca_scores, width, label='PCA', alpha=0.8, color='steelblue')
    
    # VAE
    if len(df_vae) > 0:
        vae_scores = [df_vae[df_vae['latent_dim'] == ld][metric].values[0] 
                     if len(df_vae[df_vae['latent_dim'] == ld]) > 0 else 0 
                     for ld in latent_dims]
        ax.bar(x, vae_scores, width, label='VAE', alpha=0.8, color='coral')
    
    # Best Denoised per latent dim
    denoised_scores = [df_best_denoised[df_best_denoised['latent_dim'] == ld]['score'].values[0] 
                      if len(df_best_denoised[df_best_denoised['latent_dim'] == ld]) > 0 else 0 
                      for ld in latent_dims]
    bars = ax.bar(x + width, denoised_scores, width, label='Best Denoised', alpha=0.8, color='seagreen')
    
    # Add denoise step labels on denoised bars
    for i, ld in enumerate(latent_dims):
        subset = df_best_denoised[df_best_denoised['latent_dim'] == ld]
        if len(subset) > 0:
            steps = int(subset['denoise_steps'].values[0])
            ax.text(i + width, denoised_scores[i] + 0.002, f'{steps}s', 
                   ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax.set_xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(latent_dims)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                        f'3_label_transfer_comparison_{DATA_TYPE}_{CLASSIFIER}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: label_transfer_comparison_{DATA_TYPE}_{CLASSIFIER}.png")
plt.close()

# ============================================================
# PLOT 4: Scatter plots - cLISI vs Label Transfer
# ============================================================
print("Generating correlation plots...")

combined_csv = os.path.join(RESULTS_PATH, 'label_transfer', 
                           f'combined_clisi_labeltransfer_{DATA_TYPE}_{CLASSIFIER}.csv')

if os.path.exists(combined_csv):
    df_combined = pd.read_csv(combined_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scatter_metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('test_f1_weighted', 'Test F1 (Weighted)'),
        ('test_f1_macro', 'Test F1 (Macro)'),
        ('test_balanced_accuracy', 'Test Balanced Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(scatter_metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric not in df_combined.columns:
            continue
        
        # Calculate correlation
        corr, p_value = pearsonr(df_combined['clisi_denoised'], df_combined[metric])
        
        # Scatter plot
        scatter = ax.scatter(df_combined['clisi_denoised'], df_combined[metric],
                           c=df_combined['latent_dim_int'], cmap='viridis',
                           s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Add trend line
        z = np.polyfit(df_combined['clisi_denoised'], df_combined[metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_combined['clisi_denoised'].min(), 
                            df_combined['clisi_denoised'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5, 
               label=f'Trend (r={corr:.3f}, p={p_value:.2e})')
        
        ax.set_xlabel('cLISI Score', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'cLISI vs {title}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        
        # Add colorbar (only on rightmost plots)
        if idx % 2 == 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Latent Dimension', fontsize=10, fontweight='bold')
        
        # Annotate best point
        best_idx = df_combined[metric].idxmax()
        best_row = df_combined.loc[best_idx]
        ax.annotate(f"Best: L{int(best_row['latent_dim_int'])}_S{int(best_row['denoise_steps_int'])}",
                   xy=(best_row['clisi_denoised'], best_row[metric]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2),
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                            f'4_clisi_vs_labeltransfer_{DATA_TYPE}_{CLASSIFIER}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: clisi_vs_labeltransfer_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# PLOT 5: Heatmap - Improvement over PCA
# ============================================================
print("Generating improvement heatmaps...")

if len(df_denoised) > 0 and len(df[df['type'] == 'pca']) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate improvement for each denoised config
    improvements_f1 = []
    improvements_acc = []
    
    for _, denoised_row in df_denoised.iterrows():
        latent_dim = int(denoised_row['latent_dim'])
        
        # Find corresponding PCA baseline
        pca_row = df[(df['type'] == 'pca') & (df['latent_dim'] == latent_dim)]
        
        if len(pca_row) > 0:
            pca_f1 = pca_row['test_f1_weighted'].values[0]
            pca_acc = pca_row['test_accuracy'].values[0]
            
            improvements_f1.append({
                'latent_dim': latent_dim,
                'denoise_steps': int(denoised_row['denoise_steps']),
                'improvement': denoised_row['test_f1_weighted'] - pca_f1
            })
            
            improvements_acc.append({
                'latent_dim': latent_dim,
                'denoise_steps': int(denoised_row['denoise_steps']),
                'improvement': denoised_row['test_accuracy'] - pca_acc
            })
    
    # F1 improvement
    if improvements_f1:
        df_imp_f1 = pd.DataFrame(improvements_f1)
        pivot_f1 = df_imp_f1.pivot(index='latent_dim', 
                                   columns='denoise_steps', 
                                   values='improvement')
        
        sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'F1 Improvement over PCA'},
                   linewidths=0.5, linecolor='white', ax=axes[0])
        
        axes[0].set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
        axes[0].set_title('F1 (Weighted) Improvement over PCA', fontsize=14, fontweight='bold')
    
    # Accuracy improvement
    if improvements_acc:
        df_imp_acc = pd.DataFrame(improvements_acc)
        pivot_acc = df_imp_acc.pivot(index='latent_dim', 
                                     columns='denoise_steps', 
                                     values='improvement')
        
        sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Accuracy Improvement over PCA'},
                   linewidths=0.5, linecolor='white', ax=axes[1])
        
        axes[1].set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy Improvement over PCA', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                            f'5_label_transfer_improvements_{DATA_TYPE}_{CLASSIFIER}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: label_transfer_improvements_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# PLOT 6: Summary bar plot by type
# ============================================================
print("Generating summary bar plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Group by type
type_summary = df.groupby('type').agg({
    'test_accuracy': ['mean', 'std', 'max'],
    'test_f1_weighted': ['mean', 'std', 'max']
}).reset_index()

type_order = ['pca', 'vae', 'denoised']
type_labels = ['PCA', 'VAE', 'Denoised\n(Best per Latent)']
colors_type = ['#3498db', '#f39c12', '#2ecc71']

# Test Accuracy
means_acc = []
stds_acc = []
max_acc = []
for t in type_order:
    subset = type_summary[type_summary['type'] == t]
    if len(subset) > 0:
        means_acc.append(subset[('test_accuracy', 'mean')].values[0])
        stds_acc.append(subset[('test_accuracy', 'std')].values[0])
        max_acc.append(subset[('test_accuracy', 'max')].values[0])
    else:
        means_acc.append(0)
        stds_acc.append(0)
        max_acc.append(0)

x_pos = np.arange(len(type_labels))

# Mean with error bars
bars1 = axes[0].bar(x_pos - 0.2, means_acc, 0.35, 
                    yerr=stds_acc, color=colors_type, alpha=0.8, 
                    capsize=5, label='Mean ± Std')

# Max
bars2 = axes[0].bar(x_pos + 0.2, max_acc, 0.35,
                    color=colors_type, alpha=0.5, label='Max')

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(type_labels, fontsize=11)
axes[0].set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Test Accuracy by Representation Type', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (mean, max_val) in enumerate(zip(means_acc, max_acc)):
    axes[0].text(i - 0.2, mean + stds_acc[i] + 0.005, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].text(i + 0.2, max_val + 0.005, f'{max_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Test F1 (Weighted)
means_f1 = []
stds_f1 = []
max_f1 = []
for t in type_order:
    subset = type_summary[type_summary['type'] == t]
    if len(subset) > 0:
        means_f1.append(subset[('test_f1_weighted', 'mean')].values[0])
        stds_f1.append(subset[('test_f1_weighted', 'std')].values[0])
        max_f1.append(subset[('test_f1_weighted', 'max')].values[0])
    else:
        means_f1.append(0)
        stds_f1.append(0)
        max_f1.append(0)

# Mean with error bars
bars3 = axes[1].bar(x_pos - 0.2, means_f1, 0.35,
                    yerr=stds_f1, color=colors_type, alpha=0.8,
                    capsize=5, label='Mean ± Std')

# Max
bars4 = axes[1].bar(x_pos + 0.2, max_f1, 0.35,
                    color=colors_type, alpha=0.5, label='Max')

axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(type_labels, fontsize=11)
axes[1].set_ylabel('Test F1 Score (Weighted)', fontsize=12, fontweight='bold')
axes[1].set_title('Test F1 by Representation Type', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (mean, max_val) in enumerate(zip(means_f1, max_f1)):
    axes[1].text(i - 0.2, mean + stds_f1[i] + 0.005, f'{mean:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].text(i + 0.2, max_val + 0.005, f'{max_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                        f'6_label_transfer_by_type_{DATA_TYPE}_{CLASSIFIER}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: label_transfer_by_type_{DATA_TYPE}_{CLASSIFIER}.png")
plt.close()

# ============================================================
# PLOT 7: Detailed grid view - All configs
# ============================================================
print("Generating detailed grid view...")

if len(df_denoised) > 0:
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sort for better visualization
    df_plot = df_denoised.sort_values(['latent_dim', 'denoise_steps']).reset_index(drop=True)
    df_plot['config'] = df_plot.apply(
        lambda row: f"L{int(row['latent_dim'])}_S{int(row['denoise_steps'])}", axis=1
    )
    
    x = np.arange(len(df_plot))
    width = 0.2
    
    # Plot multiple metrics
    ax.bar(x - 1.5*width, df_plot['test_accuracy'], width, 
           label='Accuracy', alpha=0.8, color='steelblue')
    ax.bar(x - 0.5*width, df_plot['test_f1_weighted'], width, 
           label='F1 (Weighted)', alpha=0.8, color='coral')
    ax.bar(x + 0.5*width, df_plot['test_f1_macro'], width, 
           label='F1 (Macro)', alpha=0.8, color='seagreen')
    ax.bar(x + 1.5*width, df_plot['test_balanced_accuracy'], width, 
           label='Balanced Acc', alpha=0.8, color='mediumpurple')
    
    ax.set_xlabel('Configuration (Latent_Steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Label Transfer Metrics Across All Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['config'], rotation=90, fontsize=8)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical separators between latent dimensions
    current_latent = None
    for i, latent in enumerate(df_plot['latent_dim']):
        if current_latent is not None and latent != current_latent:
            ax.axvline(x=i - 0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        current_latent = latent
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                            f'7_label_transfer_all_configs_{DATA_TYPE}_{CLASSIFIER}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: label_transfer_all_configs_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# PLOT 8: Effect of denoising steps (averaged over latent dims)
# ============================================================
print("Generating denoising effect plot...")
if len(df_denoised) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get VAE data (0 denoising steps, averaged over all latent dimensions)
    df_vae = df[df['type'] == 'vae'].copy()
    vae_mean_acc = df_vae['test_accuracy'].mean()
    vae_std_acc = df_vae['test_accuracy'].std()
    vae_mean_f1 = df_vae['test_f1_weighted'].mean()
    vae_std_f1 = df_vae['test_f1_weighted'].std()
    
    # Average over latent dimensions
    by_steps = df_denoised.groupby('denoise_steps').agg({
        'test_accuracy': ['mean', 'std'],
        'test_f1_weighted': ['mean', 'std']
    })
    denoise_steps = sorted(df_denoised['denoise_steps'].unique())
    
    # Add 0 step to the arrays
    denoise_steps_with_zero = [0] + denoise_steps
    
    # Test Accuracy
    means_acc = by_steps[('test_accuracy', 'mean')].values
    stds_acc = by_steps[('test_accuracy', 'std')].values
    
    # Prepend VAE values
    means_acc_with_zero = np.concatenate([[vae_mean_acc], means_acc])
    stds_acc_with_zero = np.concatenate([[vae_std_acc], stds_acc])
    
    axes[0].errorbar(denoise_steps_with_zero, means_acc_with_zero, yerr=stds_acc_with_zero,
                    marker='o', linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, color='steelblue',
                    ecolor='gray', alpha=0.8)
    
    # Highlight the 0 step (VAE) point
    axes[0].scatter([0], [vae_mean_acc], s=200, color='red', marker='s', 
                   zorder=5, edgecolors='black', linewidths=2, 
                   label='VAE (0 steps)')
    
    axes[0].set_xlabel('Denoising Steps (0 = VAE)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0].set_title('Effect of Denoising Steps on Accuracy\n(Averaged over Latent Dimensions)',
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].set_xlim(left=-5)
    
    # Add value labels
    for step, mean, std in zip(denoise_steps_with_zero, means_acc_with_zero, stds_acc_with_zero):
        axes[0].text(step, mean + std + 0.002, f'{mean:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Test F1
    means_f1 = by_steps[('test_f1_weighted', 'mean')].values
    stds_f1 = by_steps[('test_f1_weighted', 'std')].values
    
    # Prepend VAE values
    means_f1_with_zero = np.concatenate([[vae_mean_f1], means_f1])
    stds_f1_with_zero = np.concatenate([[vae_std_f1], stds_f1])
    
    axes[1].errorbar(denoise_steps_with_zero, means_f1_with_zero, yerr=stds_f1_with_zero,
                    marker='o', linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, color='coral',
                    ecolor='gray', alpha=0.8)
    
    # Highlight the 0 step (VAE) point
    axes[1].scatter([0], [vae_mean_f1], s=200, color='red', marker='s', 
                   zorder=5, edgecolors='black', linewidths=2, 
                   label='VAE (0 steps)')
    
    axes[1].set_xlabel('Denoising Steps (0 = VAE)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test F1 (Weighted) (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1].set_title('Effect of Denoising Steps on F1\n(Averaged over Latent Dimensions)',
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='best')
    axes[1].set_xlim(left=-5)
    
    # Add value labels
    for step, mean, std in zip(denoise_steps_with_zero, means_f1_with_zero, stds_f1_with_zero):
        axes[1].text(step, mean + std + 0.002, f'{mean:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer',
                            f'8_denoising_effect_{DATA_TYPE}_{CLASSIFIER}.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: denoising_effect_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# PLOT 9: Effect of latent dimensions (averaged over denoise steps)
# ============================================================
print("Generating latent dimension effect plot...")

if len(df_denoised) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average over denoising steps
    by_latent = df_denoised.groupby('latent_dim').agg({
        'test_accuracy': ['mean', 'std'],
        'test_f1_weighted': ['mean', 'std']
    })
    
    latent_dims = sorted(df_denoised['latent_dim'].unique())
    
    # Test Accuracy
    means_acc = by_latent[('test_accuracy', 'mean')].values
    stds_acc = by_latent[('test_accuracy', 'std')].values
    
    axes[0].errorbar(latent_dims, means_acc, yerr=stds_acc,
                    marker='o', linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, color='steelblue',
                    ecolor='gray', alpha=0.8)
    
    axes[0].set_xlabel('Latent Dimensions', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0].set_title('Effect of Latent Dimensions on Accuracy\n(Averaged over Denoising Steps)', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(latent_dims)
    
    # Add value labels
    for ld, mean, std in zip(latent_dims, means_acc, stds_acc):
        axes[0].text(ld, mean + std + 0.002, f'{mean:.4f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Test F1
    means_f1 = by_latent[('test_f1_weighted', 'mean')].values
    stds_f1 = by_latent[('test_f1_weighted', 'std')].values
    
    axes[1].errorbar(latent_dims, means_f1, yerr=stds_f1,
                    marker='o', linewidth=2.5, markersize=10,
                    capsize=5, capthick=2, color='coral',
                    ecolor='gray', alpha=0.8)
    
    axes[1].set_xlabel('Latent Dimensions', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test F1 (Weighted) (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1].set_title('Effect of Latent Dimensions on F1\n(Averaged over Denoising Steps)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(latent_dims)
    
    # Add value labels
    for ld, mean, std in zip(latent_dims, means_f1, stds_f1):
        axes[1].text(ld, mean + std + 0.002, f'{mean:.4f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'label_transfer', 
                            f'9_latent_dimension_effect_{DATA_TYPE}_{CLASSIFIER}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: latent_dimension_effect_{DATA_TYPE}_{CLASSIFIER}.png")
    plt.close()

# ============================================================
# Print summary statistics
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\n📊 Overall Performance:")
print(f"  Mean Test Accuracy: {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
print(f"  Mean Test F1 (weighted): {df['test_f1_weighted'].mean():.4f} ± {df['test_f1_weighted'].std():.4f}")
print(f"  Mean Test F1 (macro): {df['test_f1_macro'].mean():.4f} ± {df['test_f1_macro'].std():.4f}")

print("\n📊 By Representation Type:")
for rep_type in ['pca', 'vae', 'denoised']:
    subset = df[df['type'] == rep_type]
    if len(subset) > 0:
        print(f"\n  {rep_type.upper()}:")
        print(f"    N configs: {len(subset)}")
        print(f"    Accuracy: {subset['test_accuracy'].mean():.4f} ± {subset['test_accuracy'].std():.4f}")
        print(f"    F1 (weighted): {subset['test_f1_weighted'].mean():.4f} ± {subset['test_f1_weighted'].std():.4f}")
        print(f"    F1 (macro): {subset['test_f1_macro'].mean():.4f} ± {subset['test_f1_macro'].std():.4f}")

# Best and worst
print("\n📊 Best Configuration:")
best_idx = df['test_f1_weighted'].idxmax()
best = df.loc[best_idx]
print(f"  {best['representation']}")
print(f"  Accuracy: {best['test_accuracy']:.4f}")
print(f"  F1 (weighted): {best['test_f1_weighted']:.4f}")
print(f"  F1 (macro): {best['test_f1_macro']:.4f}")

print("\n📊 Worst Configuration:")
worst_idx = df['test_f1_weighted'].idxmin()
worst = df.loc[worst_idx]
print(f"  {worst['representation']}")
print(f"  Accuracy: {worst['test_accuracy']:.4f}")
print(f"  F1 (weighted): {worst['test_f1_weighted']:.4f}")
print(f"  F1 (macro): {worst['test_f1_macro']:.4f}")

print("\n" + "="*60)
print("✓ All label transfer plots generated!")
print("="*60)