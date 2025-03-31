library(dplyr)
library(Seurat)
library(patchwork)
library(ggplot2)

sys_info <- Sys.info()
# Check the OS and perform actions based on it
if (grepl("Linux", sys_info["sysname"])) {
  cao <- readRDS('../../../../data/gene_count_sampled.RDS')
  cao <- CreateSeuratObject(counts = cao)
  cell_annotations <- readRDS("../../../../data/df_cell.RDS")
  gene_annotations <- readRDS("../../../../data/df_gene.RDS")
} else if (sys_info["sysname"] == "Windows") {
  cao <- readRDS('C:/Users/hostp/Desktop/data/gene_count_sampled.RDS')
  cao <- CreateSeuratObject(counts = cao)
  cell_annotations <- readRDS("C:/Users/hostp/Desktop/data/df_cell.RDS")
  #gene_annotations <- readRDS("C:/Users/hostp/Desktop/data/df_gene.RDS")
}



# Load the gene mapping data from the RDS file
df_gene <- readRDS("C:/Users/hostp/Desktop/data/df_gene.RDS")


# Check the structure of the loaded file (it should have columns for Ensembl IDs and Gene Symbols)
head(df_gene)

# Ensure the 'df_gene' dataframe has two columns: EnsemblID and GeneSymbol
# If the columns are named differently, adjust accordingly
# For example, if the columns are 'ensembl' and 'symbol', rename them like this:
# colnames(df_gene) <- c("EnsemblID", "GeneSymbol")

# Map Ensembl IDs to Gene Symbols
ensembl_to_symbol <- setNames(df_gene$gene_short_name, df_gene$gene_id)

# Assuming your Seurat object is named 'cao', we want to update row names in the Seurat object to gene symbols
# First, check that the row names (gene IDs) of the Seurat object match the Ensembl IDs in df_gene
matched_genes <- rownames(cao) %in% df_gene$gene_id



# Now, map Ensembl IDs to gene symbols for the Seurat object
# Update the row names with gene symbols
rownames(cao@assays$RNA@layers$counts) <- ensembl_to_symbol[rownames(cao)]
rownames(cao@assays$RNA@layers$counts)


# Add the gene symbols as row metadata (not as cell-level metadata)
cao@assays$RNA@meta.features$gene_symbol <- ensembl_to_symbol[rownames(cao_filtered@assays$RNA@counts)]

# Verify that the gene symbols have been added correctly to the Seurat object
head(cao_filtered@assays$RNA@meta.features$gene_symbol)
