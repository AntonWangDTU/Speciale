#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Visualize grid search results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Configuration
DATA_TYPE = 'allCT_1.0k'
# If DATA argument provided OVERRIDE
if len(sys.argv) >= 2:
    DATA_TYPE = sys.argv[1]
    
RESULTS_PATH = '../results'
csv_file = os.path.join(RESULTS_PATH, f'grid_search_results_{DATA_TYPE}_final.csv')

# Load results
print(f"Loading results from: {csv_file}")
df = pd.read_csv(csv_file)
df = df.rename(columns={'clisi_raw': 'clisi_hvg'})


print(f"\nLoaded {len(df)} experiments")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================
# PLOT 1: Line plot - Denoise steps on X-axis, color by latent_dim
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['clisi_pca', 'clisi_vae', 'clisi_denoised', 'clisi_hvg']
titles = ['cLISI (PCA)', 'cLISI (VAE)', 'cLISI (Denoised)', 'cLISI (hvg)']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    if metric not in df.columns:
        ax.text(0.5, 0.5, f'{metric}\nnot available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue
    
    # Plot line for each latent dimension
    for latent_dim in sorted(df['latent_dim'].unique()):
        subset = df[df['latent_dim'] == latent_dim].sort_values('denoise_steps')
        ax.plot(subset['denoise_steps'], subset[metric], 
                marker='o', linewidth=2, markersize=8,
                label=f'Latent {int(latent_dim)}')
    
    ax.set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('cLISI Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Latent Dimension', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for latent_dim in sorted(df['latent_dim'].unique()):
        subset = df[df['latent_dim'] == latent_dim].sort_values('denoise_steps')
        for _, row in subset.iterrows():
            ax.annotate(f'{row[metric]:.3f}', 
                       xy=(row['denoise_steps'], row[metric]),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=7, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_lineplot_{DATA_TYPE}.png'), 
            dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {RESULTS_PATH}/grid_search_lineplot_{DATA_TYPE}.png")
plt.close()

# ============================================================
# PLOT 2: Heatmap for denoised cLISI
# ============================================================
if 'clisi_denoised' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot = df.pivot(index='latent_dim', 
                     columns='denoise_steps', 
                     values='clisi_denoised')
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis', 
                cbar_kws={'label': 'cLISI Score'},
                linewidths=0.5, linecolor='white',
                ax=ax)
    
    ax.set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
    ax.set_title('cLISI Scores (Denoised Latents): Grid Search Results', 
                 fontsize=14, fontweight='bold')
    
    # Highlight best configuration
    best_idx = df['clisi_denoised'].idxmax()
    best_config = df.loc[best_idx]
    best_latent = int(best_config['latent_dim'])
    best_steps = int(best_config['denoise_steps'])
    
    # Find position in pivot table
    row_idx = list(pivot.index).index(best_latent)
    col_idx = list(pivot.columns).index(best_steps)
    
    # Add red rectangle around best
    rect = plt.Rectangle((col_idx, row_idx), 1, 1, 
                         fill=False, edgecolor='red', linewidth=4)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_heatmap_{DATA_TYPE}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_PATH}/grid_search_heatmap_{DATA_TYPE}.png")
    plt.close()

# ============================================================
# PLOT 3: Comparison across metrics (grouped bar plot)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique configurations
configs = df[['latent_dim', 'denoise_steps']].drop_duplicates()

# For cleaner visualization, show only selected configurations
selected_configs = [
    (10, 50), (10, 150), (10, 250),
    (20, 50), (20, 150), (20, 250),
    (30, 50), (30, 150), (30, 250),
    (40, 50), (40, 150), (40, 250),
    (50, 50), (50, 150), (50, 250),
]

# Filter to selected configs
df_selected = df[df.apply(lambda row: (row['latent_dim'], row['denoise_steps']) in selected_configs, axis=1)]

# Create labels
df_selected['config_label'] = df_selected.apply(
    lambda row: f"L{int(row['latent_dim'])}_S{int(row['denoise_steps'])}", axis=1
)

# Prepare data for grouped bar plot
metrics_to_plot = ['clisi_pca', 'clisi_vae', 'clisi_denoised', 'clisi_hvg']
available_metrics = [m for m in metrics_to_plot if m in df_selected.columns]

x = np.arange(len(df_selected))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, metric in enumerate(available_metrics):
    offset = width * (i - len(available_metrics)/2 + 0.5)
    ax.bar(x + offset, df_selected[metric], width, 
           label=metric.replace('clisi_', '').upper(),
           color=colors[i], alpha=0.8)

ax.set_xlabel('Configuration (Latent Dimension_Denoising Steps)', fontsize=12, fontweight='bold')
ax.set_ylabel('cLISI Score', fontsize=12, fontweight='bold')
ax.set_title('cLISI Scores Across Selected Configurations', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_selected['config_label'], rotation=45, ha='right')
ax.legend(title='Metric', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_comparison_{DATA_TYPE}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_PATH}/grid_search_comparison_{DATA_TYPE}.png")
plt.close()

# ============================================================
# PLOT 4: Improvement over baseline (PCA and hvg)
# ============================================================
if 'clisi_denoised' in df.columns and 'clisi_pca' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate improvements
    df['improvement_over_pca'] = df['clisi_denoised'] - df['clisi_pca']
    df['improvement_over_hvg'] = df['clisi_denoised'] - df['clisi_hvg']
    
    # Plot 1: Improvement over PCA
    pivot_pca = df.pivot(index='latent_dim', 
                         columns='denoise_steps', 
                         values='improvement_over_pca')
    
    sns.heatmap(pivot_pca, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'cLISI Improvement'},
                linewidths=0.5, linecolor='white', ax=axes[0])
    
    axes[0].set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
    axes[0].set_title('Improvement over PCA Baseline', fontsize=14, fontweight='bold')
    
    # Plot 2: Improvement over hvg
    pivot_hvg = df.pivot(index='latent_dim', 
                         columns='denoise_steps', 
                         values='improvement_over_hvg')
    
    sns.heatmap(pivot_hvg, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'cLISI Improvement'},
                linewidths=0.5, linecolor='white', ax=axes[1])
    
    axes[1].set_xlabel('Denoising Steps', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Latent Dimensions', fontsize=12, fontweight='bold')
    axes[1].set_title('Improvement over hvg Data', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_improvements_{DATA_TYPE}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_PATH}/grid_search_improvements_{DATA_TYPE}.png")
    plt.close()

# ============================================================
# PLOT 5: Summary statistics
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Best cLISI by latent dimension (across all denoise steps)
if 'clisi_denoised' in df.columns:
    best_by_latent = df.groupby('latent_dim')['clisi_denoised'].max().reset_index()
    axes[0, 0].bar(best_by_latent['latent_dim'], best_by_latent['clisi_denoised'], 
                   color='steelblue', alpha=0.8)
    axes[0, 0].set_xlabel('Latent Dimension', fontweight='bold')
    axes[0, 0].set_ylabel('Best cLISI Score', fontweight='bold')
    axes[0, 0].set_title('Best cLISI by Latent Dimension', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for _, row in best_by_latent.iterrows():
        axes[0, 0].text(row['latent_dim'], row['clisi_denoised'], 
                       f"{row['clisi_denoised']:.4f}",
                       ha='center', va='bottom', fontweight='bold')

# Plot 2: Best cLISI by denoise steps (across all latent dims)
if 'clisi_denoised' in df.columns:
    best_by_steps = df.groupby('denoise_steps')['clisi_denoised'].max().reset_index()
    axes[0, 1].bar(best_by_steps['denoise_steps'], best_by_steps['clisi_denoised'], 
                   color='coral', alpha=0.8)
    axes[0, 1].set_xlabel('Denoising Steps', fontweight='bold')
    axes[0, 1].set_ylabel('Best cLISI Score', fontweight='bold')
    axes[0, 1].set_title('Best cLISI by Denoising Steps', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for _, row in best_by_steps.iterrows():
        axes[0, 1].text(row['denoise_steps'], row['clisi_denoised'], 
                       f"{row['clisi_denoised']:.4f}",
                       ha='center', va='bottom', fontweight='bold')

# Plot 3: Distribution of scores
available_metrics = [m for m in ['clisi_pca', 'clisi_vae', 'clisi_denoised', 'clisi_hvg'] 
                     if m in df.columns]
if available_metrics:
    data_for_box = []
    labels_for_box = []
    for metric in available_metrics:
        data_for_box.append(df[metric].values)
        labels_for_box.append(metric.replace('clisi_', '').upper())
    
    bp = axes[1, 0].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Color the boxes
    colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    axes[1, 0].set_ylabel('cLISI Score', fontweight='bold')
    axes[1, 0].set_title('Distribution of cLISI Scores by Metric', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Table with best configurations
if 'clisi_denoised' in df.columns:
    axes[1, 1].axis('off')
    
    # Get top 5 configurations
    top5 = df.nlargest(5, 'clisi_denoised')[['latent_dim', 'denoise_steps', 
                                              'clisi_pca', 'clisi_vae', 
                                              'clisi_denoised', 'clisi_hvg']]
    
    # Create table
    table_data = []
    table_data.append(['Rank', 'Latent', 'Steps', 'PCA', 'VAE', 'Denoised', 'hvg'])
    
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        table_data.append([
            str(rank),
            f"{int(row['latent_dim'])}",
            f"{int(row['denoise_steps'])}",
            f"{row.get('clisi_pca', 0):.3f}",
            f"{row.get('clisi_vae', 0):.3f}",
            f"{row['clisi_denoised']:.3f}",
            f"{row.get('clisi_hvg', 0):.3f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center',
                            bbox=[0, 0.2, 1, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best configuration
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#FFD700')
    
    axes[1, 1].set_title('Top 5 Configurations', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_summary_{DATA_TYPE}.png'), 
            dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_PATH}/grid_search_summary_{DATA_TYPE}.png")
plt.close()

# ============================================================
# PLOT 6: Individual metric focus - Denoised cLISI
# ============================================================
if 'clisi_denoised' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create color palette
    latent_dims = sorted(df['latent_dim'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(latent_dims)))
    
    for latent_dim, color in zip(latent_dims, colors):
        subset = df[df['latent_dim'] == latent_dim].sort_values('denoise_steps')
        ax.plot(subset['denoise_steps'], subset['clisi_denoised'], 
                marker='o', linewidth=3, markersize=10,
                label=f'Latent Dim {int(latent_dim)}',
                color=color)
    
    ax.set_xlabel('Denoising Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('cLISI Score (Denoised)', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Denoising Steps on cLISI Score', 
                 fontsize=16, fontweight='bold')
    ax.legend(title='Latent Dimension', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Highlight best point
    best_idx = df['clisi_denoised'].idxmax()
    best_row = df.loc[best_idx]
    ax.scatter(best_row['denoise_steps'], best_row['clisi_denoised'], 
              s=300, color='red', marker='*', zorder=5, 
              edgecolors='black', linewidths=2,
              label=f"Best: L{int(best_row['latent_dim'])}_S{int(best_row['denoise_steps'])}")
    
    ax.legend(title='Configuration', fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'grid_search_denoised_focus_{DATA_TYPE}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_PATH}/grid_search_denoised_focus_{DATA_TYPE}.png")
    plt.close()

# ============================================================
# PRINT SUMMARY STATISTICS
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

if 'clisi_denoised' in df.columns:
    print(f"\ncLISI (Denoised):")
    print(f"  Mean: {df['clisi_denoised'].mean():.4f}")
    print(f"  Std:  {df['clisi_denoised'].std():.4f}")
    print(f"  Min:  {df['clisi_denoised'].min():.4f}")
    print(f"  Max:  {df['clisi_denoised'].max():.4f}")
    
    # Best configuration
    best_idx = df['clisi_denoised'].idxmax()
    best_row = df.loc[best_idx]
    print(f"\n✓ Best Configuration:")
    print(f"  Latent Dim: {int(best_row['latent_dim'])}")
    print(f"  Denoise Steps: {int(best_row['denoise_steps'])}")
    print(f"  cLISI (PCA): {best_row.get('clisi_pca', 'N/A')}")
    print(f"  cLISI (VAE): {best_row.get('clisi_vae', 'N/A')}")
    print(f"  cLISI (Denoised): {best_row['clisi_denoised']:.4f}")
    print(f"  cLISI (hvg): {best_row.get('clisi_hvg', 'N/A')}")
    
    # Worst configuration
    worst_idx = df['clisi_denoised'].idxmin()
    worst_row = df.loc[worst_idx]
    print(f"\n✗ Worst Configuration:")
    print(f"  Latent Dim: {int(worst_row['latent_dim'])}")
    print(f"  Denoise Steps: {int(worst_row['denoise_steps'])}")
    print(f"  cLISI (Denoised): {worst_row['clisi_denoised']:.4f}")

# Effect of latent dimension
print(f"\n" + "-"*60)
print("Effect of Latent Dimension (averaged over denoise steps):")
if 'clisi_denoised' in df.columns:
    by_latent = df.groupby('latent_dim')['clisi_denoised'].agg(['mean', 'std', 'min', 'max'])
    print(by_latent.to_string())

# Effect of denoising steps
print(f"\n" + "-"*60)
print("Effect of Denoising Steps (averaged over latent dims):")
if 'clisi_denoised' in df.columns:
    by_steps = df.groupby('denoise_steps')['clisi_denoised'].agg(['mean', 'std', 'min', 'max'])
    print(by_steps.to_string())

print("\n" + "="*60)
print("✓ All plots saved!")
print("="*60)