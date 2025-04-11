library(Seurat)
library(tidyverse)
library(patchwork)
library(tidyverse)
library(ggplot2)
library(SeuratData)
InstallData("panc8")


# Get system info to determine data path
sys_info <- Sys.info()

if (grepl("Linux", sys_info["sysname"])) {
  data_dir <- '../../../data/'
} else if (sys_info["sysname"] == "Windows") {
  data_dir <- 'C:/Users/hostp/Desktop/data/'
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

panc <- LoadData('panc8')


#This code is inspired by https://satijalab.org/seurat/articles/integration_rpca.html 
#What needs to be done is to use a version of cao_subsample where the rownames has been defined for
#gene short names and aggregated for dublicates

rownames(cao_subsample) <- cao_subsample@assays$RNA@meta.data$gene_short_name
# Combine both datasets (panc and cao_subsample) into a list
seurat_list <- list(cao_subsample, panc)

common_features <- intersect(rownames(seurat_list[[1]]), rownames(seurat_list[[2]]))


# Subset both Seurat objects based on common features
seurat_list[[1]] <- subset(seurat_list[[1]], features = common_features)
seurat_list[[2]] <- subset(seurat_list[[2]], features = common_features)




# Normalize and identify variable features for each dataset independently
seurat_list <- lapply(X = seurat_list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

# Select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = seurat_list)

# Scale data and run PCA on each Seurat object
seurat_list <- lapply(X = seurat_list, FUN = function(x) {
  x <- ScaleData(x, features = features, verbose = FALSE)
  x <- RunPCA(x, features = features, verbose = FALSE)
})

# Now the list of Seurat objects (seurat_list) is ready for further analysis (integration, clust







