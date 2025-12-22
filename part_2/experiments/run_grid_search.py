#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Grid search over latent dimensions and denoising steps
Save all results for train/val/test splits
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from itertools import product
from pipeline import Pipeline
from config import VAEConfig, DiffusionConfig, ExperimentConfig
print("imported libraries")

def main():
    # Configuration
    DATA_TYPE = 'allCT_1.0k'

    # If DATA argument provided OVERRIDE
    if len(sys.argv) >= 2:
        DATA_TYPE = sys.argv[1]

    exp_config = ExperimentConfig(
        data_type=DATA_TYPE,
        data_path='../data',
        model_path='../models',
        results_path='../results',
        latent_dims=[10, 20, 30, 40, 50],
        denoise_steps=[0, 5, 10, 20, 40, 80, 160],
        random_seed=42,
        device='cuda'
    )
    
    vae_config = VAEConfig()
    diffusion_config = DiffusionConfig()

    # Initialize pipeline
    pipeline = Pipeline(exp_config, vae_config, diffusion_config)
    
    # Grid search
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Starting grid search for dataset: {DATA_TYPE}")
    print(f"Latent dimensions: {exp_config.latent_dims}")
    print(f"Denoising steps: {exp_config.denoise_steps}")
    print(f"Total configurations: {len(exp_config.latent_dims) * len(exp_config.denoise_steps)}")
    print(f"{'='*60}\n")
    
    for latent_dim, denoise_steps in product(exp_config.latent_dims, exp_config.denoise_steps):
        try:
            results = pipeline.run_single_experiment(latent_dim, denoise_steps)
            all_results.append(results)
            
            # Save intermediate results (CSV)
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(exp_config.results_path,
                                   f'grid_search_results_{DATA_TYPE}.csv'), index=False)
            
            # ========================================================
            # CHANGED: Save all three splits (not just test)
            # ========================================================
            pipeline.adata_train.write_h5ad(
                os.path.join(exp_config.data_path,
                           f'{DATA_TYPE}_grid_search_train.h5ad')
            )
            pipeline.adata_val.write_h5ad(
                os.path.join(exp_config.data_path,
                           f'{DATA_TYPE}_grid_search_val.h5ad')
            )
            pipeline.adata_test_master.write_h5ad(
                os.path.join(exp_config.data_path,
                           f'{DATA_TYPE}_grid_search_test.h5ad')
            )
            
            print(f"\n✓ Saved consolidated results to AnnData (train/val/test)")
            print(f"  Train representations: {len(pipeline.adata_train.obsm)}")
            print(f"  Val representations: {len(pipeline.adata_val.obsm)}")
            print(f"  Test representations: {len(pipeline.adata_test_master.obsm)}")
            
        except Exception as e:
            print(f"\n❌ Error in experiment (latent_dim={latent_dim}, denoise_steps={denoise_steps}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================
    # CONSOLIDATION: Final summary and saves
    # ============================================================
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    
    df = pd.DataFrame(all_results)
    print(df.to_string())
    
    # Save final results CSV
    df.to_csv(os.path.join(exp_config.results_path,
                           f'grid_search_results_{DATA_TYPE}_final.csv'), index=False)
    
    # Save final consolidated AnnData for all three splits
    final_train_path = os.path.join(exp_config.data_path,
                                    f'{DATA_TYPE}_grid_search_all_results_train.h5ad')
    final_val_path = os.path.join(exp_config.data_path,
                                  f'{DATA_TYPE}_grid_search_all_results_val.h5ad')
    final_test_path = os.path.join(exp_config.data_path,
                                   f'{DATA_TYPE}_grid_search_all_results_test.h5ad')
    
    pipeline.adata_train.write_h5ad(final_train_path)
    pipeline.adata_val.write_h5ad(final_val_path)
    pipeline.adata_test_master.write_h5ad(final_test_path)
    
    print(f"\n✓ Final consolidated AnnData saved:")
    print(f"  Train: {final_train_path}")
    print(f"  Val:   {final_val_path}")
    print(f"  Test:  {final_test_path}")
    
    # ========================================================
    # CHANGED: Show representations for all splits
    # ========================================================
    print(f"\n✓ Available representations:")
    print(f"  Train: {pipeline.adata_train.n_obs} cells, {len(pipeline.adata_train.obsm)} representations")
    print(f"  Val:   {pipeline.adata_val.n_obs} cells, {len(pipeline.adata_val.obsm)} representations")
    print(f"  Test:  {pipeline.adata_test_master.n_obs} cells, {len(pipeline.adata_test_master.obsm)} representations")
    
    #print(f"\n✓ Detailed representation list:")
    #for key in sorted(pipeline.adata_test_master.obsm.keys()):
    #    shape_train = pipeline.adata_train.obsm[key].shape if key in pipeline.adata_train.obsm else "N/A"
    #    shape_val = pipeline.adata_val.obsm[key].shape if key in pipeline.adata_val.obsm else "N/A"
    #    shape_test = pipeline.adata_test_master.obsm[key].shape
    #    print(f"  - {key}")
    #    print(f"      Train: {shape_train}, Val: {shape_val}, Test: {shape_test}")
    
    # Find best configuration
    if 'clisi_denoised' in df.columns and len(df) > 0:
        best_idx = df['clisi_denoised'].idxmax()
        best_config = df.loc[best_idx]
        
        print("\n" + "="*60)
        print("BEST CONFIGURATION:")
        print(f"Dataset: {DATA_TYPE}")
        print(f"Latent Dim: {int(best_config['latent_dim'])}")
        print(f"Denoise Steps: {int(best_config['denoise_steps'])}")
        print(f"cLISI (PCA): {best_config.get('clisi_pca', 'N/A')}")
        print(f"cLISI (VAE): {best_config.get('clisi_vae', 'N/A')}")
        print(f"cLISI (Denoised): {best_config['clisi_denoised']:.4f}")
        print(f"cLISI (Raw): {best_config.get('clisi_raw', 'N/A')}")
        print("="*60)
        sys.exit(0)

if __name__ == "__main__":
    main()