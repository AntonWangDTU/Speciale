#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Run single experiment with specific configuration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import Pipeline
from config import VAEConfig, DiffusionConfig, ExperimentConfig


def main():
    # If DATA argument provided OVERRIDE
    if len(sys.argv) >= 2:
        DATA_TYPE = sys.argv[1]
    
    # Configuration
    exp_config = ExperimentConfig(
        data_type=DATA_TYPE,
        data_path='../data',
        model_path='../models',
        results_path='../results',
        random_seed=42,
        device='cuda'
    )
    
    vae_config = VAEConfig()
    diffusion_config = DiffusionConfig()
    
    # Initialize pipeline
    pipeline = Pipeline(exp_config, vae_config, diffusion_config)
    
    # Run single experiment
    latent_dim = 40
    denoise_steps = 30
    
    results = pipeline.run_single_experiment(latent_dim, denoise_steps)
    
    print("\n" + "="*60)
    print(f"Experiment complete: Latent Dim={latent_dim}, Denoise Steps={denoise_steps}")
    print("="*60)


if __name__ == "__main__":
    main()  