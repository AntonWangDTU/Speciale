#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Dataset classes for VAE and Diffusion training
"""
import torch
from torch.utils.data import Dataset


class scRNADataset(Dataset):
    """Dataset for raw scRNA-seq data (VAE training)"""
    def __init__(self, adata):
        self.data = torch.FloatTensor(
            adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        )
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]


class LatentDataset(Dataset):
    """Dataset for latent vectors (Diffusion training)"""
    def __init__(self, latent_vectors):
        """
        latent_vectors: torch.Tensor of shape (N, latent_dim)
        """
        self.latents = latent_vectors
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx]