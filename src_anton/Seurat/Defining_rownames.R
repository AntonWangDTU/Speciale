
library(dplyr)
library(Seurat)
library(patchwork)
library(ggplot2)


# This script takes the cao data, subsamples it and then collapses the dublicate gene short names,
# so that there are not dublicates present. It also renames the rownames of the seurat object

# Get the system information
sys_info <- Sys.info()
if (grepl("Linux", sys_info["sysname"])) {
  data_dir = '../../../data/'
} else if (sys_info["sysname"] == "Windows") {
  data_dir = 'C:/Users/hostp/Desktop/data/'
}
# Path to saved Seurat object
seurat_path <- file.path(data_dir, "cao_subsample.rds")

# Check if the object already exists
if (file.exists(seurat_path)) {
  message("Loading existing Seurat object...")
  cao_subsample <- readRDS(seurat_path) 
} else {
  print("Run cao exploration, to create seurat object")
}



# Assuming rownames of your Seurat object are Ensembl IDs with version
ensembl_ids <- rownames(cao_subsample)

# Create named vector of gene symbols
ens_to_symbol <- setNames(cao_subsample@assays$RNA@meta.data$gene_short_name, cao_subsample@assays$RNA@meta.data$gene_id)

# Map gene symbols to Seurat object rownames
gene_symbols <- ens_to_symbol[ensembl_ids]

# Assign as new rownames
rownames(cao_subsample) <- gene_symbols


# Check for duplicates
sum(duplicated(rownames(cao_subsample)))

# View duplicated gene names
dup_genes <- rownames(cao_subsample)[duplicated(rownames(cao_subsample))]
head(dup_genes)


###
###This part agregates the counts for dublicate genes
###

library(Matrix)

# Get raw counts
counts <- GetAssayData(cao_subsample, layer = "counts")

# Aggregate by gene name
counts_agg <- rowsum(as.matrix(counts), group = rownames(cao_subsample))

# Rebuild Seurat object from aggregated data
cao_subsample_agg <- CreateSeuratObject(counts = counts_agg)


# Check for duplicates
sum(duplicated(rownames(cao_subsample_agg)))


seurat_path <- file.path(data_dir, "cao_subsample.rds")
saveRDS(cao_subsample_agg, seurat_path)
message("Seurat object saved to: ", seurat_path)

