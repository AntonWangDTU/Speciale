#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
VAE Model for scRNA-seq data with ZINB distribution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class scRNAVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, hidden_dims=[100], dropout_rate=0.2):
        """
        VAE for log-normalized HVG input
        """
        super(scRNAVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # REconstruction head: back to HVG space
        self.fc_recon = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        x_recon = self.fc_recon(h)
        return x_recon

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z
    

    def get_latent(self, x):
        """Get latent representation without reparameterization"""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
def vae_loss(x, x_recon, mu, logvar, kl_weight=1.0):
    """Combined VAE loss: reconstruction + KL divergence"""
    recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss +kl_weight*kl_loss
    return total_loss, recon_loss, kl_loss

