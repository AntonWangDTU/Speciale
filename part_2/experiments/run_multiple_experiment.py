#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Run multiple runs experiment with specific configuration
"""
import sys
import json
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import Pipeline
from config import VAEConfig, DiffusionConfig, ExperimentConfig


def main():
    # If DATA argument provided OVERRIDE
    if len(sys.argv) == 2:
        DATA_TYPE = sys.argv[1]
        n_runs = 10
    elif len(sys.argv) >= 3:
        DATA_TYPE = sys.argv[1]
        n_runs = int(sys.argv[2]) 
    else:
        print("Error no data name or n_runs provided:")
        print("Correct command: run_multiple_experiments.py (datatype) optional(n_runs)")
        sys.exit(1)


    # HYPERPARAMS: PUT IN OPTIMAL HYPERPARAMS FOR DATA
    # Load hyperparameters for a single dataset
    dataset_name = DATA_TYPE
    print(f"Importing grid search parameters from:  /novo/users/iwaq/multi/project_{dataset_name}/results/best_hyperparameters.json")
    with open(f'/novo/users/iwaq/multi_alt/project_{dataset_name}/results/best_hyperparameters.json', 'r') as f:
        params = json.load(f)

    # Access the values
    latent_dim = params['best_denoised']['latent_dim']
    denoise_steps = params['best_denoised']['denoise_steps']
    latent_dim_pca = params['best_pca']['latent_dim']


    print(f"Using best vae/LDM latentdim and steps: {latent_dim} and {denoise_steps}")
    print(f"Using best PCA dim: {latent_dim_pca}")

    #latent_dim = 30 #VAE and VAE+LDM
    #denoise_steps = 20
    #latent_dim_pca = 50

    # NO of runs
    #n_runs = 10

    # Empty results list:
    results_list = []


    for i in range(n_runs): 
        # Configuration
        exp_config = ExperimentConfig(
            data_type=DATA_TYPE,
            data_path='../data',
            model_path='../models',
            results_path='../results',
            random_seed=i, # RANDOM SEED FOR EACH RUN 
            device='cuda'
        )
        
        vae_config = VAEConfig()
        diffusion_config = DiffusionConfig()
        
        # Initialize pipeline
        pipeline = Pipeline(exp_config, vae_config, diffusion_config)

        # Results for run   
        results = pipeline.run_single_experiment(latent_dim, denoise_steps, latent_dim_pca)
        # Add data col
        results["data_type"] = DATA_TYPE

        # Append to list
        results_list.append(results)
        
        print("\n" + "="*60)
        print(f"Experiment complete: Latent Dim={latent_dim}, Denoise Steps={denoise_steps}")
        print("="*60)

    # Convert to DataFrame at the end
    results_df = pd.DataFrame(results_list)
    print("\nFinal Results:")
    print(results_df)

    # Save final results CSV
    results_df.to_csv(os.path.join(exp_config.results_path,
                        f'multiple_run_{DATA_TYPE}_nruns_{n_runs}.csv'), index=False)


if __name__ == "__main__":
    main()  
