#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Main pipeline for VAE training, diffusion denoising, and evaluation
"""
import os
import torch
import torch.nn.functional as F
import scanpy as sc
import numpy as np
import pandas as pd
import scib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vae import scRNAVAE, vae_loss
from models.diffusion import GaussianDiffusion, LatentDenoiser
from utils.dataset import scRNADataset, LatentDataset
from utils.evaluation import compute_clisi_metrics
from config import VAEConfig, DiffusionConfig, ExperimentConfig


class Pipeline:
    """Complete pipeline for VAE + Diffusion + Evaluation"""
    
    def __init__(self, exp_config: ExperimentConfig,
                 vae_config: VAEConfig,
                 diffusion_config: DiffusionConfig):
        self.exp_config = exp_config
        self.vae_config = vae_config
        self.diffusion_config = diffusion_config
        
        # Set random seeds
        torch.manual_seed(exp_config.random_seed)
        np.random.seed(exp_config.random_seed)
        
        # Setup device
        self.device = torch.device(exp_config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(exp_config.model_path, exist_ok=True)
        os.makedirs(exp_config.results_path, exist_ok=True)
    
    def load_data(self, filename):
        """Load and split data"""
        print(f"Loading data from {filename}...")
        adata = sc.read_h5ad(filename)
        
        # Split data
        train_idx, temp_idx = train_test_split(
            np.arange(adata.n_obs), test_size=0.3, random_state=self.exp_config.random_seed,
            stratify=adata.obs['cell_type'] if 'cell_type' in adata.obs else None
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.33, random_state=self.exp_config.random_seed,
            stratify=adata.obs.iloc[temp_idx]['cell_type'] if 'cell_type' in adata.obs else None
        )
        
        adata_train = adata[train_idx].copy()
        adata_val = adata[val_idx].copy()
        adata_test = adata[test_idx].copy()
        
        print(f"Train: {adata_train.n_obs}, Val: {adata_val.n_obs}, Test: {adata_test.n_obs}")
        
        return adata_train, adata_val, adata_test, adata
    
    def train_vae(self, adata_train, adata_val, latent_dim):
        """Train VAE model"""
        print(f"\n{'='*60}")
        print(f"Training VAE with latent dimension: {latent_dim}")
        print(f"{'='*60}")
        
        # Check for NaN in input data
        X_train = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
        if np.isnan(X_train).any():
            nan_count = np.isnan(X_train).sum()
            print(f"❌ ERROR: Found {nan_count} NaN values in training data!")
            raise ValueError("Input data contains NaN values!")
        
        print(f"✓ Input data validated (no NaN values)")
        print(f"  Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"  Data mean: {X_train.mean():.2f}") 
        # Create datasets
        train_dataset = scRNADataset(adata_train)
        val_dataset = scRNADataset(adata_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.vae_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.vae_config.batch_size, shuffle=False)
        
        # Initialize model
        model = scRNAVAE(
            input_dim=adata_train.n_vars,
            latent_dim=latent_dim,
            hidden_dims=self.vae_config.hidden_dims,
            dropout_rate=self.vae_config.dropout_rate
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.vae_config.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.vae_config.max_epochs):
            # Train
            model.train()
            train_loss = 0
            train_recon = 0
            train_kl = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Check for NaN in input
                if torch.isnan(batch).any():
                    #print(f"Warning: NaN detected in input batch at epoch {epoch+1}")
                    continue
                
                optimizer.zero_grad()
                
                x_recon, mu_z, logvar_z = model(batch)
                
                # Check for NaN in model outputs
                if torch.isnan(x_recon).any():
                    #print(f"Warning: NaN in model outputs at epoch {epoch+1}, skipping batch")
                    continue
                
                kl_weight = min(1.0, epoch / 100)
                loss, recon_loss, kl_loss = vae_loss(batch, x_recon, mu_z, logvar_z, kl_weight)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
            
            train_loss /= len(train_loader)
            train_recon /= len(train_loader)
            train_kl /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            val_recon = 0
            val_kl = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    if torch.isnan(batch).any():
                        continue
                    
                    x_recon, mu_z, logvar_z = model(batch)
                    loss, recon_loss, kl_loss = vae_loss(batch, x_recon, mu_z, logvar_z, kl_weight)
                    
                    if torch.isnan(loss):
                        continue
                    
                    val_loss += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
            
            val_loss /= len(val_loader)
            val_recon /= len(val_loader)
            val_kl /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.vae_config.max_epochs} | "
                      f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            
            # Check for NaN in losses
            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"\n❌ NaN detected in losses at epoch {epoch+1}. Stopping training.")
                break
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path =  os.path.join(self.exp_config.model_path, f'vae_latent{latent_dim}_seed{self.exp_config.random_seed}_best.pt')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.vae_config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model_path = os.path.join(self.exp_config.model_path, f'vae_latent{latent_dim}_seed{self.exp_config.random_seed}_best.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'VAE Training (Latent Dim: {latent_dim})')
        plt.savefig(os.path.join(self.exp_config.results_path, f'vae_latent{latent_dim}_training.png'))
        plt.close()
        
        return model
    
    def extract_latents(self, model, adata_list):
        """Extract latent representations from VAE"""
        model.eval()
        latent_list = []
        
        for adata in adata_list:
            dataset = scRNADataset(adata)
            with torch.no_grad():
                latents = model.get_latent(
                    torch.FloatTensor(dataset.data).to(self.device)
                ).cpu().numpy()
            
            # ============================================================
            # NaN CHECK (STEP 2) - ADD THIS SECTION
            # ============================================================
            if np.isnan(latents).any():
                nan_count = np.isnan(latents).sum()
                total_values = latents.size
                print(f"⚠️  WARNING: Found {nan_count}/{total_values} NaN values in latents!")
                print(f"   Percentage: {100 * nan_count / total_values:.2f}%")
                
                # Raise error and force retraining with better parameters
                raise ValueError(
                    "NaN values detected in latent representations! "
                    "This indicates VAE training failed. "
                    "Please delete cached models and retrain VAE with lower learning rate."
                )
            # ============================================================
            
            latent_list.append(latents)
        
        return latent_list 

    def train_diffusion(self, latents_train, latent_dim):
        """Train diffusion model"""
        print(f"\n{'='*60}")
        print(f"Training Diffusion Model (Latent Dim: {latent_dim})")
        print(f"{'='*60}")
        
        # Create dataset
        latents_tensor = torch.from_numpy(latents_train).float()
        dataset = LatentDataset(latents_tensor)
        dataloader = DataLoader(dataset, batch_size=self.diffusion_config.batch_size,
                              shuffle=True, num_workers=4)
        
        # Initialize model
        model = LatentDenoiser(
            latent_dim=latent_dim,
            n_steps=self.diffusion_config.timesteps,
            hidden_dim=self.diffusion_config.hidden_dim,
            time_dim=self.diffusion_config.time_dim,
            num_layers=self.diffusion_config.num_layers,
            position_embedding=self.diffusion_config.position_embedding
        ).to(self.device)
        
        # Initialize diffusion process
        diffusion = GaussianDiffusion(
            timesteps=self.diffusion_config.timesteps,
            beta_schedule=self.diffusion_config.beta_schedule,
            device=self.device
        )
        
        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.diffusion_config.learning_rate)
        
        # EMA setup
        shadow_params = None
        if self.diffusion_config.use_ema:
            shadow_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    shadow_params[name] = param.data.clone()
        
        model.train()
        for epoch in range(self.diffusion_config.num_epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.diffusion_config.num_epochs}', miniters=10)
            
            for batch in pbar:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=self.device).long()
                noise = torch.randn_like(batch)
                noisy_latents = diffusion.q_sample(batch, t, noise=noise)
                
                predicted_noise = model(noisy_latents, t)
                loss = F.mse_loss(predicted_noise, noise)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                
                # Update EMA
                if self.diffusion_config.use_ema:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                shadow_params[name].mul_(self.diffusion_config.ema_decay).add_(
                                    param.data, alpha=1.0 - self.diffusion_config.ema_decay
                                )
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}')
        
        # Apply EMA weights
        if self.diffusion_config.use_ema:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(shadow_params[name])
        
        # Save model
        model_path = os.path.join(self.exp_config.model_path, f'diffusion_latent{latent_dim}_seed{self.exp_config.random_seed}.pt')
        torch.save(model.state_dict(), model_path)
        
        return model, diffusion
    
    def denoise_latents(self, diffusion_model, diffusion, latents, denoise_steps):
        """Denoise latents using trained diffusion model"""
        diffusion_model.eval()
        
        latents_tensor = torch.from_numpy(latents).float()
        dataset = LatentDataset(latents_tensor)
        dataloader = DataLoader(dataset, batch_size=self.diffusion_config.batch_size, shuffle=False)
        
        denoised_batches = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Denoising (steps={denoise_steps})"):
                batch = batch.to(self.device)
                denoised = diffusion.denoise(diffusion_model, batch, self.device,
                                            denoise_steps=denoise_steps, verbose=False)
                denoised_batches.append(denoised.cpu())
        
        denoised_latents = torch.cat(denoised_batches, dim=0).numpy()
        return denoised_latents
    
    def compute_metrics(self, adata_test, vae_key, denoised_key, pca_key):
        """Compute cLISI metrics for specific representations"""
        metrics = {}
        
        # PCA
        if pca_key in adata_test.obsm:
            metrics['clisi_pca'] = scib.metrics.clisi_graph(
                adata_test, label_key="cell_type", type_="embed",
                use_rep=pca_key, k0=90, subsample=None,
                scale=True, n_cores=1, verbose=False
            )
        
        # VAE latents
        if vae_key in adata_test.obsm:
            metrics['clisi_vae'] = scib.metrics.clisi_graph(
                adata_test, label_key="cell_type", type_="embed",
                use_rep=vae_key, k0=90, subsample=None,
                scale=True, n_cores=1, verbose=False
            )
        
        # Denoised latents
        if denoised_key in adata_test.obsm:
            metrics['clisi_denoised'] = scib.metrics.clisi_graph(
                adata_test, label_key="cell_type", type_="embed",
                use_rep=denoised_key, k0=90, subsample=None,
                scale=True, n_cores=1, verbose=False
            )
        
        # Raw (only once)
        if not hasattr(self, '_clisi_raw_computed'):
            metrics['clisi_raw'] = scib.metrics.clisi_graph(
                self.adata_test_processed, label_key="cell_type", type_="full",
                k0=90, subsample=None, scale=True,
                n_cores=1, verbose=False
            )
            self._clisi_raw_computed = True
            self._clisi_raw_value = metrics['clisi_raw']
        else:
            metrics['clisi_raw'] = self._clisi_raw_value
        
        return metrics
    
    def train_vae_for_latent_dim(self, latent_dim):
        """
        Train VAE and extract latents for a given latent dimension.
        This should only be called ONCE per latent_dim.
        
        Returns:
            tuple: (vae_model, latents_train, latents_val, latents_test)
        """
        # Check if already trained
        model_path = os.path.join(self.exp_config.model_path, f'vae_latent{latent_dim}_seed{self.exp_config.random_seed}_best.pt')
        latents_cache_path = os.path.join(self.exp_config.model_path, f'vae_latent{latent_dim}_seed{self.exp_config.random_seed}_latents.npz')
        
        if os.path.exists(model_path) and os.path.exists(latents_cache_path):
            print(f"\n{'='*60}")
            print(f"✓ Found existing VAE model for latent_dim={latent_dim}")
            print(f"  Loading cached model and latents...")
            print(f"{'='*60}")
            
            # Load model
            vae_model = scRNAVAE(
                input_dim=self.adata_train.n_vars,
                latent_dim=latent_dim,
                hidden_dims=self.vae_config.hidden_dims,
                dropout_rate=self.vae_config.dropout_rate
            ).to(self.device)
            vae_model.load_state_dict(torch.load(model_path))
            vae_model.eval()
            
            # Load cached latents
            latents_cache = np.load(latents_cache_path)
            latents_train = latents_cache['latents_train']
            latents_val = latents_cache['latents_val']
            latents_test = latents_cache['latents_test']
            
            print(f"  ✓ Loaded VAE model and cached latents")
            
            return vae_model, latents_train, latents_val, latents_test
        
        # Train VAE
        vae_model = self.train_vae(self.adata_train, self.adata_val, latent_dim)
        
        # Extract latents
        print("\nExtracting VAE latents...")
        latents_train, latents_val, latents_test = self.extract_latents(
            vae_model, [self.adata_train, self.adata_val, self.adata_test_master]
        )
        
        # Cache latents for future use
        np.savez_compressed(
            latents_cache_path,
            latents_train=latents_train,
            latents_val=latents_val,
            latents_test=latents_test
        )
        print(f"✓ Cached latents to: {latents_cache_path}")
        
        return vae_model, latents_train, latents_val, latents_test
    
    def train_diffusion_for_config(self, latents_train, latent_dim, denoise_steps):
        """
        Train diffusion model for a specific configuration.
        
        Args:
            latents_train: Training latents from VAE
            latent_dim: Latent dimension
            denoise_steps: Number of denoising steps
        
        Returns:
            tuple: (diffusion_model, diffusion_process)
        """
        # Check if already trained
        diffusion_model_path = os.path.join(
            self.exp_config.model_path,
            f'diffusion_latent{latent_dim}_seed{self.exp_config.random_seed}.pt'
            )
        
        if os.path.exists(diffusion_model_path):
            print(f"\n{'='*60}")
            print(f"✓ Found existing diffusion model for latent_dim={latent_dim}")
            print(f"  Loading cached model...")
            print(f"{'='*60}")
            
            # Load model
            diffusion_model = LatentDenoiser(
                latent_dim=latent_dim,
                n_steps=self.diffusion_config.timesteps,
                hidden_dim=self.diffusion_config.hidden_dim,
                time_dim=self.diffusion_config.time_dim,
                num_layers=self.diffusion_config.num_layers,
                position_embedding=self.diffusion_config.position_embedding
            ).to(self.device)
            diffusion_model.load_state_dict(torch.load(diffusion_model_path))
            diffusion_model.eval()
            
            # Initialize diffusion process
            diffusion = GaussianDiffusion(
                timesteps=self.diffusion_config.timesteps,
                beta_schedule=self.diffusion_config.beta_schedule,
                device=self.device
            )
            
            print(f"  ✓ Loaded diffusion model")
            
            return diffusion_model, diffusion
        
        # Train diffusion
        diffusion_model, diffusion = self.train_diffusion(latents_train, latent_dim)
        return diffusion_model, diffusion

    def evaluate_label_transfer(self, latent_dim_pca, latent_dim, denoise_steps, classifier_type='logistic_regression'):

        """
        Evaluate label transfer performance for current configuration
        
        Args:
            latent_dim: Current latent dimension
            denoise_steps: Current denoising steps
            classifier_type: 'logistic_regression' or 'random_forest'
        
        Returns:
            dict: Label transfer metrics for all representations
        """
        print("\n" + "="*60)
        print("EVALUATING LABEL TRANSFER")
        print("="*60)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        # Helper function to get classifier
        def get_classifier(clf_type):
            if clf_type == 'logistic_regression':
                return LogisticRegression(
                    max_iter=1000,
                    random_state=self.exp_config.random_seed,
                    n_jobs=-1,
                    solver='lbfgs',
                    multi_class='multinomial'
                )
            elif clf_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.exp_config.random_seed,
                    n_jobs=-1,
                    max_depth=20
                )
            else:
                raise ValueError(f"Unknown classifier: {clf_type}")
        
        # Get labels from all splits
        y_train = self.adata_train.obs['cell_type'].values
        y_val = self.adata_val.obs['cell_type'].values
        y_test = self.adata_test_master.obs['cell_type'].values
        
        # Dictionary to store all metrics
        label_transfer_metrics = {}
        
        # Add dataset statistics
        label_transfer_metrics['n_cells_train'] = self.adata_train.n_obs
        label_transfer_metrics['n_cells_val'] = self.adata_val.n_obs
        label_transfer_metrics['n_cells_test'] = self.adata_test_master.n_obs
        label_transfer_metrics['n_cell_types'] = len(np.unique(y_train))
        label_transfer_metrics['n_genes'] = self.adata_train.n_vars
        
        print(f"\nDataset Statistics:")
        print(f"  Train cells: {label_transfer_metrics['n_cells_train']}")
        print(f"  Val cells: {label_transfer_metrics['n_cells_val']}")
        print(f"  Test cells: {label_transfer_metrics['n_cells_test']}")
        print(f"  Cell types: {label_transfer_metrics['n_cell_types']}")
        print(f"  Genes: {label_transfer_metrics['n_genes']}")
        
        # ============================================================
        # 1. Evaluate HVG Preprocessed
        # ============================================================
        print("\n1. Evaluating HVG preprocessed data...")
        
        # Get HVG data (already preprocessed and stored)
        X_train_hvg = self.adata_train_processed.X.toarray() if hasattr(self.adata_train_processed.X, 'toarray') else self.adata_train_processed.X
        X_test_hvg = self.adata_test_processed.X.toarray() if hasattr(self.adata_test_processed.X, 'toarray') else self.adata_test_processed.X
        
        try:
            clf = get_classifier(classifier_type)
            clf.fit(X_train_hvg, y_train)
            
            y_test_pred = clf.predict(X_test_hvg)
            
            label_transfer_metrics['lt_hvg_test_acc'] = accuracy_score(y_test, y_test_pred)
            label_transfer_metrics['lt_hvg_test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
            
            print(f"  ✓ HVG - Test Acc: {label_transfer_metrics['lt_hvg_test_acc']:.4f}, F1: {label_transfer_metrics['lt_hvg_test_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ HVG failed: {e}")
            label_transfer_metrics['lt_hvg_test_acc'] = np.nan
            label_transfer_metrics['lt_hvg_test_f1'] = np.nan
        
        # ============================================================
        # 2. Evaluate PCA
        # ============================================================
        print(f"\n2. Evaluating PCA (latent_dim={latent_dim_pca})...")
        
        pca_key = f'X_pca_latent{latent_dim_pca}'
        
        try:
            X_train_pca = self.adata_train.obsm[pca_key]
            X_test_pca = self.adata_test_master.obsm[pca_key]
            
            clf = get_classifier(classifier_type)
            clf.fit(X_train_pca, y_train)
            
            y_test_pred = clf.predict(X_test_pca)
            
            label_transfer_metrics['lt_pca_test_acc'] = accuracy_score(y_test, y_test_pred)
            label_transfer_metrics['lt_pca_test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
            
            print(f"  ✓ PCA - Test Acc: {label_transfer_metrics['lt_pca_test_acc']:.4f}, F1: {label_transfer_metrics['lt_pca_test_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ PCA failed: {e}")
            label_transfer_metrics['lt_pca_test_acc'] = np.nan
            label_transfer_metrics['lt_pca_test_f1'] = np.nan
        
        # ============================================================
        # 3. Evaluate VAE
        # ============================================================
        print(f"\n3. Evaluating VAE (latent_dim={latent_dim})...")
        
        vae_key = f'X_VAE_latent{latent_dim}'
        
        try:
            X_train_vae = self.adata_train.obsm[vae_key]
            X_test_vae = self.adata_test_master.obsm[vae_key]
            
            clf = get_classifier(classifier_type)
            clf.fit(X_train_vae, y_train)
            
            y_test_pred = clf.predict(X_test_vae)
            
            label_transfer_metrics['lt_vae_test_acc'] = accuracy_score(y_test, y_test_pred)
            label_transfer_metrics['lt_vae_test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
            
            print(f"  ✓ VAE - Test Acc: {label_transfer_metrics['lt_vae_test_acc']:.4f}, F1: {label_transfer_metrics['lt_vae_test_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ VAE failed: {e}")
            label_transfer_metrics['lt_vae_test_acc'] = np.nan
            label_transfer_metrics['lt_vae_test_f1'] = np.nan
        
        # ============================================================
        # 4. Evaluate Denoised
        # ============================================================
        print(f"\n4. Evaluating Denoised (latent_dim={latent_dim}, denoise_steps={denoise_steps})...")
        
        denoised_key = f'X_VAE_denoised_latent{latent_dim}_steps{denoise_steps}'
        
        try:
            X_train_denoised = self.adata_train.obsm[denoised_key]
            X_test_denoised = self.adata_test_master.obsm[denoised_key]
            
            clf = get_classifier(classifier_type)
            clf.fit(X_train_denoised, y_train)
            
            y_test_pred = clf.predict(X_test_denoised)
            
            label_transfer_metrics['lt_denoised_test_acc'] = accuracy_score(y_test, y_test_pred)
            label_transfer_metrics['lt_denoised_test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
            
            print(f"  ✓ Denoised - Test Acc: {label_transfer_metrics['lt_denoised_test_acc']:.4f}, F1: {label_transfer_metrics['lt_denoised_test_f1']:.4f}")
        except Exception as e:
            print(f"  ✗ Denoised failed: {e}")
            label_transfer_metrics['lt_denoised_test_acc'] = np.nan
            label_transfer_metrics['lt_denoised_test_f1'] = np.nan
        
        print("\n" + "="*60)
        print("Label Transfer Evaluation Complete")
        print("="*60)
        
        return label_transfer_metrics

    def run_single_experiment(self, latent_dim, denoise_steps, latent_dim_pca = None):
        """Run complete pipeline for one configuration"""
        print(f"\n{'#'*60}")
        print(f"# Experiment: Latent Dim={latent_dim}, Denoise Steps={denoise_steps}")
        print(f"{'#'*60}")
        
        # Load data (only once for first experiment)
        if not hasattr(self, 'adata_test_master'):
            data_file = os.path.join(self.exp_config.data_path,
                                    f'{self.exp_config.data_type}.h5ad')
            print("Loading and splitting data (first time only)...")
            adata_train, adata_val, adata_test, adata_full = self.load_data(data_file)
            self.adata_train = adata_train
            self.adata_val = adata_val
            self.adata_test_master = adata_test.copy()
            self.adata_full = adata_full
        
        # Use references to the master copies
        adata_train = self.adata_train
        adata_val = self.adata_val
        adata_test = self.adata_test_master
        
        # if no latent_dim for pca
        latent_dim_pca = latent_dim if latent_dim_pca is None else latent_dim_pca
        
        # ============================================================
        # STEP 0: Preprocess and identify HVGs FIRST (moved up)
        # ============================================================
        if not hasattr(self, 'adata_train_processed'):
            print("\nPreprocessing all splits and identifying HVGs...")
            # Train
            adata_train_processed = adata_train.copy()
            sc.pp.normalize_total(adata_train_processed, target_sum=1e4)
            sc.pp.log1p(adata_train_processed)
            
            # Identify HVGs from training set
            batch_key = "sample" if "sample" in adata_train_processed.obs.columns else None
            if batch_key:
                sc.pp.highly_variable_genes(adata_train_processed, n_top_genes=2000,
                                        batch_key=batch_key, subset=False)
            else:
                sc.pp.highly_variable_genes(adata_train_processed, n_top_genes=2000, subset=False)
            
            # Get HVG names
            hvg_genes = adata_train_processed.var_names[adata_train_processed.var['highly_variable']].tolist()
            self.hvg_genes = hvg_genes  # Store for later use
            
            # Subset to HVGs
            adata_train_processed = adata_train_processed[:, hvg_genes].copy()
            
            # Val (use same HVGs from training)
            adata_val_processed = adata_val.copy()
            sc.pp.normalize_total(adata_val_processed, target_sum=1e4)
            sc.pp.log1p(adata_val_processed)
            adata_val_processed = adata_val_processed[:, hvg_genes].copy()
            
            # Test (use same HVGs from training)
            adata_test_processed = adata_test.copy()
            sc.pp.normalize_total(adata_test_processed, target_sum=1e4)
            sc.pp.log1p(adata_test_processed)
            adata_test_processed = adata_test_processed[:, hvg_genes].copy()
            
            # Store processed versions
            self.adata_train_processed = adata_train_processed
            self.adata_val_processed = adata_val_processed
            self.adata_test_processed = adata_test_processed
            print(f"  ✓ Preprocessed data shape: {adata_train_processed.shape}")
            print(f"  ✓ Number of HVGs: {len(hvg_genes)}")
        
        # ============================================================
        # STEP 1: Train VAE on HVGs (or load cached) - ONCE per latent_dim
        # ============================================================
        # MODIFIED: Temporarily replace self.adata_train, val, test with HVG versions
        # Save original references
        _original_train = self.adata_train
        _original_val = self.adata_val
        _original_test = self.adata_test_master
        
        # Temporarily replace with HVG versions for VAE training
        self.adata_train = self.adata_train_processed
        self.adata_val = self.adata_val_processed
        self.adata_test_master = self.adata_test_processed
        
        # Call existing train_vae_for_latent_dim (it will now use HVG data)
        vae_model, latents_train, latents_val, latents_test = self.train_vae_for_latent_dim(latent_dim)
        
        # Restore original references
        self.adata_train = _original_train
        self.adata_val = _original_val
        self.adata_test_master = _original_test
        
        # Store VAE latents in ALL splits (original anndata objects)
        vae_key = f'X_VAE_latent{latent_dim}'
        adata_train.obsm[vae_key] = latents_train
        adata_val.obsm[vae_key] = latents_val
        adata_test.obsm[vae_key] = latents_test
        
        # ============================================================
        # STEP 2: Train Diffusion (or load cached) - ONCE per latent_dim
        # ============================================================
        diffusion_model, diffusion = self.train_diffusion_for_config(
            latents_train, latent_dim, denoise_steps
        )
        
        # ============================================================
        # STEP 3: Denoise with specific step count - UNIQUE per config
        # ============================================================
        print(f"\nDenoising latents with {denoise_steps} steps...")
        # Denoise ALL three splits
        denoised_key = f'X_VAE_denoised_latent{latent_dim}_steps{denoise_steps}'
        
        # Check cache for all splits
        denoised_cache_train = os.path.join(
            self.exp_config.model_path,
            f'denoised_latent{latent_dim}_steps{denoise_steps}_seed{self.exp_config.random_seed}_train.npz'
        )
        denoised_cache_val = os.path.join(
            self.exp_config.model_path,
            f'denoised_latent{latent_dim}_steps{denoise_steps}_seed{self.exp_config.random_seed}_val.npz'
        )
        denoised_cache_test = os.path.join(
            self.exp_config.model_path,
            f'denoised_latent{latent_dim}_steps{denoise_steps}_seed{self.exp_config.random_seed}_test.npz'
        )
        
        # Denoise train
        if os.path.exists(denoised_cache_train):
            print(f"  ✓ Loading cached denoised latents (train)...")
            denoised_train = np.load(denoised_cache_train)['denoised']
        else:
            print(f"  Denoising training set...")
            denoised_train = self.denoise_latents(
                diffusion_model, diffusion, latents_train, denoise_steps
            )
            np.savez_compressed(denoised_cache_train, denoised=denoised_train)
        
        # Denoise val
        if os.path.exists(denoised_cache_val):
            print(f"  ✓ Loading cached denoised latents (val)...")
            denoised_val = np.load(denoised_cache_val)['denoised']
        else:
            print(f"  Denoising validation set...")
            denoised_val = self.denoise_latents(
                diffusion_model, diffusion, latents_val, denoise_steps
            )
            np.savez_compressed(denoised_cache_val, denoised=denoised_val)
        
        # Denoise test
        if os.path.exists(denoised_cache_test):
            print(f"  ✓ Loading cached denoised latents (test)...")
            denoised_test = np.load(denoised_cache_test)['denoised']
        else:
            print(f"  Denoising test set...")
            denoised_test = self.denoise_latents(
                diffusion_model, diffusion, latents_test, denoise_steps
            )
            np.savez_compressed(denoised_cache_test, denoised=denoised_test)
        
        # Store denoised latents in ALL splits
        adata_train.obsm[denoised_key] = denoised_train
        adata_val.obsm[denoised_key] = denoised_val
        adata_test.obsm[denoised_key] = denoised_test
        
        # ============================================================
        # STEP 4: Compute PCA on HVGs (once per latent_dim) on ALL splits
        # ============================================================
        pca_key = f'X_pca_latent{latent_dim_pca}'
        if pca_key not in adata_test.obsm:
            print(f"\nComputing PCA on HVGs for latent dimension {latent_dim_pca}...")
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Use HVG-preprocessed data instead of full data
            X_train = self.adata_train_processed.X.toarray() if hasattr(self.adata_train_processed.X, 'toarray') else self.adata_train_processed.X
            X_val = self.adata_val_processed.X.toarray() if hasattr(self.adata_val_processed.X, 'toarray') else self.adata_val_processed.X
            X_test = self.adata_test_processed.X.toarray() if hasattr(self.adata_test_processed.X, 'toarray') else self.adata_test_processed.X
            
            # Fit scaler and PCA on train only
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            pca = PCA(n_components=latent_dim_pca, random_state=self.exp_config.random_seed)
            pca.fit(X_train_scaled)
            
            # Transform ALL splits
            adata_train.obsm[pca_key] = pca.transform(X_train_scaled)
            adata_val.obsm[pca_key] = pca.transform(X_val_scaled)
            adata_test.obsm[pca_key] = pca.transform(X_test_scaled)
            
            print(f"  ✓ PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
            print(f"  ✓ PCA computed on {X_train.shape[1]} HVGs")
        
        # ============================================================
        # STEP 5: Evaluate cLISI (only on test set for metrics)
        # ============================================================
        print("\nComputing cLISI metrics on test set...")
        clisi_metrics = self.compute_metrics(adata_test, vae_key, denoised_key, pca_key)
        
        print("\n" + "="*50)
        print("cLISI Results:")
        for key, value in clisi_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("="*50)
        
        # ============================================================
        # STEP 6: Evaluate Label Transfer
        # ============================================================
        label_transfer_metrics = self.evaluate_label_transfer(latent_dim_pca, latent_dim, denoise_steps)
        
        # ============================================================
        # Combine all results
        # ============================================================
        results = {
            'latent_dim': latent_dim,
            'denoise_steps': denoise_steps,
            **clisi_metrics,
            **label_transfer_metrics
        }
        
        print("\n" + "="*60)
        print("COMBINED RESULTS SUMMARY")
        print("="*60)
        print("\nDataset Info:")
        print(f"  Cells (train/val/test): {results['n_cells_train']}/{results['n_cells_val']}/{results['n_cells_test']}")
        print(f"  Cell types: {results['n_cell_types']}")
        print(f"  Genes (HVGs): {len(self.hvg_genes)}")
        
        print("\ncLISI Metrics:")
        for key in ['clisi_pca', 'clisi_vae', 'clisi_denoised', 'clisi_hvg']:
            if key in results:
                print(f"  {key}: {results[key]:.4f}")
        
        print("\nLabel Transfer (Test Set):")
        for method in ['hvg', 'pca', 'vae', 'denoised']:
            acc_key = f'lt_{method}_test_acc'
            f1_key = f'lt_{method}_test_f1'
            if acc_key in results and not np.isnan(results[acc_key]):
                print(f"  {method.upper():10s} - Acc: {results[acc_key]:.4f}, F1: {results[f1_key]:.4f}")
        print("="*60)
        
        return results
