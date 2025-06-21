#!/zhome/1a/4/136247/miniconda3/envs/ms/bin/python3.10


import scanpy as sc


adata = sc.read_h5ad('../../../blackhole/data/TabulaSapiens.h5ad', backed='r')


