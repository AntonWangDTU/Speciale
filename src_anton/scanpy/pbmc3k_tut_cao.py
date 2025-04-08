#!/home/aws/miniconda3/bin/python3
import os
import platform
import scanpy as sc
import pandas as pd

# Check the system type and set the data directory
sys_info = platform.system()
if sys_info == "Linux":
    data_dir = '../../../data/'
elif sys_info == "Windows":
    data_dir = 'C:/Users/hostp/Desktop/data/'

# Load the RDS files (in Scanpy, this would typically be .h5ad format, but we assume you're converting RDS to .h5ad or CSV)
cao_data = pd.read_csv(os.path.join(
    data_dir, "gene_count_sampled.csv"), index_col=0)
cell_annotations = pd.read_csv(os.path.join(
    data_dir, "df_cell.csv"), index_col=0)
gene_annotations = pd.read_csv(os.path.join(
    data_dir, "df_gene.csv"), index_col=0)
