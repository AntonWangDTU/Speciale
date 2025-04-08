
library(Seurat)
library(patchwork)
library(ggplot2)
library(tidyverse)



# Get system info to determine data path
sys_info <- Sys.info()

if (grepl("Linux", sys_info["sysname"])) {
  data_dir <- '../../../data/'
} else if (sys_info["sysname"] == "Windows") {
  data_dir <- 'C:/Users/hostp/Desktop/data/'
}

# Path to saved Seurat object
seurat_path <- file.path(data_dir, "cao_seurat.rds")

# Check if the object already exists
message("Loading existing Seurat object...")
cao <- readRDS(seurat_path)




glimpse(test)










