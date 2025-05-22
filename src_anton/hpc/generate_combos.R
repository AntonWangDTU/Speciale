#!/usr/bin/env Rscript

# generate_combos.R
# This script creates a list of Organ_Group_Organ_Celltype combinations

# Load required library
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(dplyr))

# Path to your Seurat RDS file

# Load Seurat object
obj <- readRDS("C:/Users/hostp/Desktop/data/tabula/tabula_RDS/tabula_preproccesed.rds")

# Ensure metadata columns exist
obj@meta.data <- obj@meta.data %>%
  rename(
    Organ    = organ,
    Celltype = cell_type
  ) %>%
  mutate(cell_id = rownames(.))

# Clean up Organ names (remove underscores)
obj$Organ <- gsub("_", " ", obj$Organ)

# Define Organ_Group mapping
to_group <- c(
  # Digestive
  Liver             = "Digestive",
  Pancreas          = "Digestive",
  Small_Intestine   = "Digestive",
  Large_Intestine   = "Digestive",
  Salivary_Gland    = "Digestive",
  Tongue            = "Digestive",
  
  # Respiratory and Immune
  Lung              = "RespiratoryImmune",
  Trachea           = "RespiratoryImmune",
  Spleen            = "RespiratoryImmune",
  Thymus            = "RespiratoryImmune",
  Lymph_Node        = "RespiratoryImmune",
  Blood             = "RespiratoryImmune",
  Bone_Marrow       = "RespiratoryImmune",
  
  # Reproductive and Endocrine
  Uterus            = "ReproductiveEndocrine",
  Placenta          = "ReproductiveEndocrine",
  Mammary           = "ReproductiveEndocrine",
  Prostate          = "ReproductiveEndocrine",
  Adrenal           = "ReproductiveEndocrine",
  
  # Urinary and Cardiovascular
  Kidney            = "UroCardiac",
  Bladder           = "UroCardiac",
  Heart             = "UroCardiac",
  Vasculature       = "UroCardiac",
  
  # Musculoskeletal
  Muscle            = "Musculoskeletal",
  Fat               = "Musculoskeletal",
  Skin              = "Musculoskeletal",
  
  # Neural and Sensory
  Eye               = "NeuralSensory"
)

# Assign Organ_Group
obj@meta.data$Organ_Group <- to_group[obj@meta.data$Organ]

# Build unique combinations
combos <- unique(with(obj@meta.data,
                      paste(Organ_Group, Organ, Celltype, sep = "_")
))

# Write combos to file
default_file <- "Speciale/src_anton/hpc/combo_list.txt"
writeLines(combos, default_file)
message("Generated ", length(combos), " combos and wrote to ", default_file)
