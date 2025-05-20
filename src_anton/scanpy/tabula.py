import scanpy as sc

adata = sc.read('/home/aws/data/4de7ab62-475c-43f9-a1d4-bfd693e7df07/TabulaSapiens.h5ad', backed='r')

print(f"Shape: {adata.shape}")
print(f"First 5 cell names: {adata.obs_names[:5]}")
