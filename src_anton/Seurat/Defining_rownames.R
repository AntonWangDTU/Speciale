
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

cao_data <- readRDS(file.path(data_dir, "gene_count_sampled.RDS"))
cell_annotations <- readRDS(file.path(data_dir, "df_cell.RDS"))
gene_annotations <- readRDS(file.path(data_dir, "df_gene.RDS"))

cao <- CreateSeuratObject(counts = cao_data, project = "cao", meta.data = cell_annotations)

#Add gene ids to RNA@meta.data
cao[["RNA"]] <- AddMetaData(cao[["RNA"]], metadata = gene_annotations)


#Make subsample
cao_subsample <- subset(cao, cells = sample(Cells(cao), 10000))

# Assuming rownames of your Seurat object are Ensembl IDs with version
ensembl_ids <- rownames(cao_subsample)

# Create named vector of gene symbols
ens_to_symbol <- setNames(gene_annotations$gene_short_name, gene_annotations$gene_id)

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



