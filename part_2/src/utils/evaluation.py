#!/novo/users/iwaq/.conda/envs/myenv/bin/python
"""
Evaluation metrics for latent representations
"""
import scib


def compute_clisi_metrics(adata, label_key="cell_type", k0=90):
    """
    Compute cLISI scores for different representations
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with different representations
    label_key : str
        Column name for cell type labels
    k0 : int
        Number of neighbors for cLISI computation
    
    Returns:
    --------
    dict : Dictionary with cLISI scores
    """
    results = {}
    
    # PCA
    if 'X_pca' in adata.obsm:
        results['clisi_pca'] = scib.metrics.clisi_graph(
            adata, label_key=label_key, type_="embed",
            use_rep="X_pca", k0=k0, subsample=None,
            scale=True, n_cores=1, verbose=False
        )
    
    # VAE latents
    if 'X_VAE' in adata.obsm:
        results['clisi_vae'] = scib.metrics.clisi_graph(
            adata, label_key=label_key, type_="embed",
            use_rep="X_VAE", k0=k0, subsample=None,
            scale=True, n_cores=1, verbose=False
        )
    
    # Denoised latents
    if 'X_VAE_denoised' in adata.obsm:
        results['clisi_denoised'] = scib.metrics.clisi_graph(
            adata, label_key=label_key, type_="embed",
            use_rep="X_VAE_denoised", k0=k0, subsample=None,
            scale=True, n_cores=1, verbose=False
        )
    
    # Raw data
    results['clisi_raw'] = scib.metrics.clisi_graph(
        adata, label_key=label_key, type_="full",
        k0=k0, subsample=None, scale=True,
        n_cores=1, verbose=False
    )
    
    return results