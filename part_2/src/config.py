#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Configuration file for experiments
"""
from dataclasses import dataclass
from typing import List


@dataclass
class VAEConfig:
    """VAE training configuration"""
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    batch_size: int = 128
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 45
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [100]


@dataclass
class DiffusionConfig:
    """Diffusion model configuration"""
    timesteps: int = 2000
    beta_schedule: str = 'sigmoid'
    hidden_dim: int = 512
    time_dim: int = 256
    num_layers: int = 4
    position_embedding: str = 'learned'
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-4
    use_ema: bool = True
    ema_decay: float = 0.9999



@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    data_type: str 
    data_path: str = '../data'
    model_path: str = './models'
    results_path: str = './results'
    latent_dims: List[int] = None
    denoise_steps: List[int] = None
    random_seed: int = 42
    device: str = 'cuda'
    
    def __post_init__(self):
        if self.latent_dims is None:
            self.latent_dims = [10, 20, 30, 40, 50]
        if self.denoise_steps is None:
            self.denoise_steps = [50, 100, 150, 200, 250]