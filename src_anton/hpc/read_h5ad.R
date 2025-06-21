#!/usr/bin/env Rscript

# ---------------------- User settings ----------------------
# Path to input .h5ad
input_h5ad   <-      "C:/Users/hostp/Desktop/data/tabula/TabulaSapiens.h5ad"

# Directory to dump the BPCells matrix
bp_matrix_dir <-     "C:/Users/hostp/Desktop/data/tabula/tabula_BP"  

# Path to save final Seurat object
output_seurat_rds <- "C:/Users/hostp/Desktop/data/tabula/tabula_RDS/tabula_preproccesed.rds"
# ------------------------------------------------------------

# Load libraries
library(hdf5r)
library(BPCells)
library(Seurat)
library(SeuratObject)
library(SeuratDisk)

# Helper to decode categorical obs columns in .h5ad
decode_obs_column <- function(h5, column_name) {
  idx    <- h5[[paste0("obs/", column_name)]]$read()
  labels <- h5[[paste0("obs/__categories/", column_name)]]$read()
  labels[idx + 1]
}

# 1) Read metadata from .h5ad
h5 <- H5File$new(input_h5ad, mode = "r")
meta <- data.frame(
  cell_id   =    h5[["obs/cell_id"]]$read(),
  cell_type = decode_obs_column(h5, "cell_ontology_class"),
  organ     = decode_obs_column(h5, "organ_tissue"),
  method    = decode_obs_column(h5, "method"),
  stringsAsFactors = FALSE
)
h5$close_all()

# 2) Extract and dump the expression matrix to disk
mat <- open_matrix_anndata_hdf5(input_h5ad)
write_matrix_dir(mat = mat, dir = bp_matrix_dir)

# 3) Reload matrix and create Seurat object
counts <- open_matrix_dir(dir = bp_matrix_dir)
so <- CreateSeuratObject(counts = counts, meta.data = meta)

# 4) Basic QC & filtering
so[["percent.mt"]] <- PercentageFeatureSet(so, pattern = "^MT-")
so <- subset(
  so, 
  subset = nFeature_RNA > 200 & nFeature_RNA < 7500 & percent.mt < 5 & method == "10X"
)
so$method <- NULL  # drop unused column

# 5) Normalize & clean up
so <- NormalizeData(so, normalization.method = "LogNormalize")
so@meta.data <- so@meta.data[ , setdiff(colnames(so@meta.data), c("orig.ident","cell_id","percent.mt")) ]

# 6) Save final Seurat object
saveRDS(so, file = output_seurat_rds)

message("All done! Seurat object saved to: ", output_seurat_rds)
