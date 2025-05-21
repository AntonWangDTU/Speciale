#!/usr/bin/env Rscript
# run_tabula_pipeline.R

# 1) CONFIG & LIBRARIES ------------------------------------------------------
Sys.setenv(OPENAI_API_KEY = 'sk-jjP-Nd86f06aJD4VBJWc_w')
BASE_URL           <- "https://api.marketplace.novo-genai.com/v1/chat/completions"
MODEL              <- "openai_gpt4_turbo_128k"
TOP_N_GENES        <- 10
MAX_RETRIES        <- 2
MIN_CELLS_PER_GROUP<- 30
MARKER_LOG_DIR     <- "marker_logs"
PREDICTION_LOG_FILE<- "predictions_combined.csv"

library(Seurat)
library(hdf5r)
library(BPCells)
library(tidyverse)
library(httr)
library(jsonlite)
library(future)
library(furrr)

source("Speciale/src_anton/tabula/functions.R")

# 2) LOAD & PREP SEURAT OBJECT ------------------------------------------------
obj <- readRDS("C:/Users/hostp/Desktop/data/tabula/tabula_RDS/tabula_preproccesed.rds")

# Rename & subset as in your script
obj@meta.data <- obj@meta.data %>%
  rename(Organ = organ, Celltype = cell_type) %>%
  mutate(cell_id = rownames(.))

obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")
obj <- subset(obj, subset = nFeature_RNA > 200 & nFeature_RNA < 7500 & percent.mt < 5)

# Add Organ_Group & Organ_Celltype
# Add Organ_Group column
organ_to_group <- c(
  # Digestive system
  Liver = "Digestive",
  Pancreas = "Digestive",
  Small_Intestine = "Digestive",
  Large_Intestine = "Digestive",
  Salivary_Gland = "Digestive",
  Tongue = "Digestive",

  # Respiratory and Immune systems
  Lung = "RespiratoryImmune",
  Trachea = "RespiratoryImmune",
  Spleen = "RespiratoryImmune",
  Thymus = "RespiratoryImmune",
  Lymph_Node = "RespiratoryImmune",
  Blood = "RespiratoryImmune",
  Bone_Marrow = "RespiratoryImmune",

  # Reproductive and Endocrine
  Uterus = "ReproductiveEndocrine",
  Placenta = "ReproductiveEndocrine",
  Mammary = "ReproductiveEndocrine",
  Prostate = "ReproductiveEndocrine",
  Adrenal = "ReproductiveEndocrine",

  # Urinary and Cardiovascular
  Kidney = "UroCardiac",
  Bladder = "UroCardiac",
  Heart = "UroCardiac",
  Vasculature = "UroCardiac",

  # Musculoskeletal
  Muscle = "Musculoskeletal",
  Fat = "Musculoskeletal",
  Skin = "Musculoskeletal",

  # Neural and Sensory
  Eye = "NeuralSensory"
)
obj@meta.data$Organ_Group <- organ_to_group[obj@meta.data$Organ]
obj$Organ <- gsub("_", " ", obj$Organ)
obj$Organ_Celltype <- paste(obj$Organ, obj$Celltype, sep = "_")
Idents(obj) <- "Organ_Celltype"

# 3) BUILD LIST OF COMBOS -----------------------------------------------------
combos_df <- obj@meta.data %>%
  distinct(Organ_Group, Organ, Celltype, Organ_Celltype)
# A tibble of rows: each Organ_Celltype with its group & organ

# 4) PARALLEL MARKER+PREDICT --------------------------------------------------
plan(multisession, workers = parallel::detectCores() - 1)

process_row <- function(row) {
  group     <- row$Organ_Group
  organ     <- row$Organ
  cell_type <- row$Celltype
  combo     <- row$Organ_Celltype
  
  # subset objects
  data_all   <- obj
  data_group <- subset(obj, Organ_Group == group)
  data_organ <- subset(obj, Organ == organ)
  
  # run the full pipeline for this combo
  pred_df <- run_prediction_for_celltype(
    data_all, data_group, data_organ,
    group, organ, cell_type
  )
  return(pred_df)
}

message("Computing markers & GPT predictions in parallel…")
predictions_list <- future_map(
  split(combos_df, seq(nrow(combos_df))),
  process_row,
  .progress = TRUE
)
predictions_df <- bind_rows(predictions_list)
write_csv(predictions_df, MARKER_LOG_DIR %>% paste0("/", basename(PREDICTION_LOG_FILE)))

# 5) SPLIT & PARALLEL EVALUATION ----------------------------------------------
chunks <- split(
  predictions_df,
  (seq_len(nrow(predictions_df)) - 1) %/% 10
)

message("Evaluating predictions in parallel…")
results_list <- future_map(
  chunks,
  ~ evaluate_prediction_llm(.x, model = MODEL, api_key = Sys.getenv("OPENAI_API_KEY"), base_url = BASE_URL),
  .progress = TRUE
)
results <- bind_rows(results_list)
saveRDS(results, "results_tabula.rds")

# 6) REPORT & PLOT ------------------------------------------------------------
compute_match_accuracy(results)

eval <- results
eval_summary <- eval %>%
  mutate(match = factor(match, levels = c("No Match","Partial Match","Match"))) %>%
  pivot_longer(Condition:Query, names_to="CategoryType", values_to="Category") %>%
  count(CategoryType, Category, match) %>%
  group_by(CategoryType, Category) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

category_order <- eval_summary %>%
  filter(match=="Match") %>%
  arrange(desc(prop)) %>%
  pull(Category) %>%
  unique()

eval_summary$Category <- factor(eval_summary$Category, levels=category_order)

library(ggplot2)
ggplot(eval_summary, aes(Category, prop, fill=match)) +
  geom_col(position="fill") +
  scale_y_continuous(labels=scales::percent) +
  facet_wrap(~CategoryType, scales="free_x", ncol=1) +
  theme_minimal() +
  theme(axis.text.x=element_text(angle=45,hjust=1))

message("Pipeline complete!")
