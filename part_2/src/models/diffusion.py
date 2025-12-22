#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Diffusion Model for denoising latent representations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ============= Noise Schedules =============
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoidal_beta_schedule(timesteps, start=1e-5, end=1e-2):
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (end - start) + start
    return torch.clip(betas, 0.0001, 0.9999)


# ============= Diffusion Process =============
class GaussianDiffusion:
    def __init__(self, timesteps=2000, beta_schedule='sigmoid', device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoidal_beta_schedule(timesteps)
        
        self.betas = betas.to(device)
        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, device):
        """Reverse diffusion - single denoising step"""
        betas_t = self.betas[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1)
        
        predicted_noise = model(x_t, t)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def denoise(self, model, noisy_latents, device, denoise_steps=None, verbose=True):
        """
        Refine clean latent vectors x0 by:
            1) Adding noise at time t = denoise_steps
            2) Running reverse deffisuion from t ->0
        """
        if denoise_steps is None:
            denoise_steps = self.timesteps // 2
        
        b = noisy_latents.shape[0]
        x0 = noisy_latents
        
        # ------------------------------------------------------------
        # (1) Forward Noising
        # ------------------------------------------------------------
        t_forward = torch.full((b,), denoise_steps, device=device, dtype=torch.long)
        x = self.q_sample(x0, t_forward)


        # ------------------------------------------------------------
        # (2) Reverse Denoising
        # ------------------------------------------------------------
        iterator = tqdm(reversed(range(0, denoise_steps)), desc='Denoising', total=denoise_steps) if verbose else reversed(range(0, denoise_steps))
        
        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, device)
        
        return x


# ============= Denoising Network =============
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LatentDenoiser(nn.Module):
    """MLP-based denoiser for latent vectors"""
    def __init__(self, latent_dim, n_steps, hidden_dim=512, time_dim=256, num_layers=4,
                 position_embedding='sinusoidal'):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.position_embedding = position_embedding
        self.num_layers = num_layers
        
        if position_embedding == 'sinusoidal':
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            
            layers = []
            layers.append(nn.Linear(latent_dim + time_dim, hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.Linear(hidden_dim, latent_dim))
            self.net = nn.Sequential(*layers)
            
        elif position_embedding == 'learned':
            self.step_embeddings = nn.ModuleList([
                nn.Embedding(n_steps, hidden_dim)
                for _ in range(num_layers - 1)
            ])
            
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(latent_dim, hidden_dim))
            self.linears.append(nn.ReLU())
            
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
                self.linears.append(nn.ReLU())
            
            self.linears.append(nn.Linear(hidden_dim, latent_dim))
    
    def forward(self, x, t):
        if self.position_embedding == 'sinusoidal':
            t_emb = self.time_mlp(t)
            x = torch.cat([x, t_emb], dim=-1)
            return self.net(x)
        elif self.position_embedding == 'learned':
            for idx, embedding_layer in enumerate(self.step_embeddings):
                t_embedding = embedding_layer(t)
                x = self.linears[2 * idx](x)
                x = x + t_embedding
                x = self.linears[2 * idx + 1](x)
            x = self.linears[-1](x)
            return x