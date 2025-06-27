#!/home/aws/miniconda3/bin/python3


import scanpy as sc

adata = sc.read_h5ad(
    '/home/aws/Documents/data/vieira19_Nasal_anonymised.processed.h5ad')
