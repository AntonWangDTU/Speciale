#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
combo <- args[1]

# Config
library(Seurat)
library(tidyverse)
library(httr)
library(jsonlite)

source("functions.R")

# Set API key
Sys.setenv(OPENAI_API_KEY = "your_key_here")

# Load full object
obj <- readRDS("tabula_preproccesed.rds")

# Preprocess (same as your setup code)
obj@meta.data <- obj@meta.data %>%
  rename(Organ = organ, Celltype = cell_type) %>%
  mutate(cell_id = rownames(.))

obj$Organ <- gsub("_", " ", obj$Organ)
obj$Organ_Celltype <- paste(obj$Organ, obj$Celltype, sep = "_")
Idents(obj) <- "Organ_Celltype"

# Extract group, organ, celltype
parts <- unlist(strsplit(combo, "_"))
group <- parts[1]
organ <- parts[2]
celltype <- parts[3]

# Skip if already processed
marker_file <- file.path("marker_logs", paste0(combo, "_markers.csv"))
if (file.exists(marker_file)) {
  message("Already processed: ", combo)
  quit(save = "no")
}

# Subset data
group_data <- subset(obj, Organ_Group == group)
organ_data <- subset(group_data, Organ == organ)

# Run prediction
pred <- run_prediction_for_celltype(obj, group_data, organ_data, group, organ, celltype)

# Save output
write_csv(pred, "predictions_combined.csv", append = file.exists("predictions_combined.csv"))
