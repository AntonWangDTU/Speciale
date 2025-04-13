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

# Combine both datasets (panc and cao_subsample) into a list
seurat_list <- list(cao_subsample, panc)

common_features <- intersect(rownames(seurat_list[[1]]), rownames(seurat_list[[2]]))


# Subset both Seurat objects based on common features
seurat_list[[1]] <- subset(seurat_list[[1]], features = common_features)
seurat_list[[2]] <- subset(seurat_list[[2]], features = common_features)


for (x in seurat_list) {
  print(length(rownames(x)))
}

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


immune.anchors <- FindIntegrationAnchors(object.list = seurat_list, anchor.features = features, reduction = "rpca")

immune.combined <- IntegrateData(anchorset = immune.anchors, k.weight = 20)

# specify that we will perform downstream analysis on the corrected data note that the
# original unmodified data still resides in the 'RNA' assay
DefaultAssay(immune.combined) <- "integrated"

# Run the standard workflow for visualization and clustering
immune.combined <- ScaleData(immune.combined, verbose = FALSE)
immune.combined <- RunPCA(immune.combined, npcs = 30, verbose = FALSE)
immune.combined <- RunUMAP(immune.combined, reduction = "pca", dims = 1:30)
immune.combined <- FindNeighbors(immune.combined, reduction = "pca", dims = 1:30)
immune.combined <- FindClusters(immune.combined, resolution = 0.5)



p1 <- DimPlot(immune.combined, reduction = "umap", group.by = "stim")
p2 <- DimPlot(immune.combined, reduction = "umap", group.by = "seurat_annotations", label = TRUE,
              repel = TRUE)
p1 + p2


