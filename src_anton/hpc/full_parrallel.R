#!/usr/bin/env Rscript
#SBATCH --job-name=DE_loop
#SBATCH --output=logs/DE_loop_%A_%a.out
#SBATCH --error=logs/DE_loop_%A_%a.err
#SBATCH --array=1-1000        # will be reset below to the real number of combos
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# === 1) ENVIRONMENT & CONFIG ===
Sys.setenv(OPENAI_API_KEY = 'sk-jjP-Nd86f06aJD4VBJWc_w')
OPENAI_API_KEY <- Sys.getenv("OPENAI_API_KEY")
BASE_URL <- "https://api.marketplace.novo-genai.com/v1/chat/completions"
MODEL <- "openai_gpt4_turbo_128k"
TOP_N_GENES <- 10
MAX_RETRIES <- 2
MIN_CELLS_PER_GROUP <- 30
MARKER_LOG_DIR <- "marker_logs"
PREDICTION_LOG_FILE <- "predictions_combined.csv"

library(Seurat)
library(tidyverse)
library(httr)
library(jsonlite)
source("functions.R")

# === 2) LOAD & PREPARE ===
obj <- readRDS("C:/Users/hostp/Desktop/data/tabula/tabula_RDS/tabula_preproccesed.rds")
md <- obj@meta.data %>%
  rename(Organ = organ, Celltype = cell_type) %>%
  mutate(cell_id = rownames(.))
obj@meta.data <- md

# filter small groups
obj <- subset(obj, cells = {
  md2 <- obj@meta.data %>% group_by(Organ, Celltype) %>% filter(n() >= MIN_CELLS_PER_GROUP)
  md2$cell_id
})

# add Organ_Group
organ_to_group <- c(
  Liver="Digestive", Pancreas="Digestive", Small_Intestine="Digestive", Large_Intestine="Digestive",
  Salivary_Gland="Digestive", Tongue="Digestive",
  Lung="RespiratoryImmune", Trachea="RespiratoryImmune", Spleen="RespiratoryImmune",
  Thymus="RespiratoryImmune", Lymph_Node="RespiratoryImmune", Blood="RespiratoryImmune",
  Bone_Marrow="RespiratoryImmune",
  Uterus="ReproductiveEndocrine", Placenta="ReproductiveEndocrine", Mammary="ReproductiveEndocrine",
  Prostate="ReproductiveEndocrine", Adrenal="ReproductiveEndocrine",
  Kidney="UroCardiac", Bladder="UroCardiac", Heart="UroCardiac", Vasculature="UroCardiac",
  Muscle="Musculoskeletal", Fat="Musculoskeletal", Skin="Musculoskeletal",
  Eye="NeuralSensory"
)
obj@meta.data$Organ_Group <- organ_to_group[obj@meta.data$Organ]
obj$Organ <- gsub("_", " ", obj$Organ)
obj$Organ_Celltype <- paste(obj$Organ, obj$Celltype, sep = "_")
Idents(obj) <- "Organ_Celltype"

# === 3) BUILD COMBO TABLE ===
combos <- obj@meta.data %>%
  distinct(Organ_Group, Organ, Celltype) %>%
  arrange(Organ_Group, Organ, Celltype) %>%
  mutate(combo_name = paste(Organ_Group, Organ, Celltype, sep = "_")) %>%
  as_tibble()

# reset array max to real n
ncombos <- nrow(combos)
if (as.integer(Sys.getenv("SLURM_ARRAY_TASK_MAX")) != ncombos) {
  # re-submit with correct array length
  message("Your combo count is ", ncombos,
          "; please re-run: sbatch --array=1-", ncombos, " run_DE_array.R")
  quit(status = 1)
}

# === 4) PICK YOUR SLURM INDEX ===
task <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
if (is.na(task) || task < 1 || task > ncombos) {
  stop("Invalid SLURM_ARRAY_TASK_ID: ", task)
}
this <- combos[task, ]
group    <- this$Organ_Group
organ    <- this$Organ
celltype <- this$Celltype
combo    <- this$combo_name

cat("Running combo #", task, "/", ncombos, " => ", combo, "\n", sep="")

# skip if done
if (!dir.exists(MARKER_LOG_DIR)) dir.create(MARKER_LOG_DIR)
done <- list.files(MARKER_LOG_DIR, pattern = "_markers.csv") %>%
  str_remove("_markers.csv")
if (combo %in% done) {
  cat("✅ Already done. Exiting.\n"); quit(status = 0)
}

# === 5) SUBSET & RUN ===
data_group <- subset(obj, Organ_Group == group)
data_organ <- subset(data_group, Organ == organ)
pred <- run_prediction_for_celltype(obj, data_group, data_organ, group, organ, celltype)

# === 6) WRITE OUT ===
write_csv(pred, file.path(MARKER_LOG_DIR, paste0(combo, "_markers.csv")))
write_csv(pred, PREDICTION_LOG_FILE, append = file.exists(PREDICTION_LOG_FILE))

cat("✅ Finished ", combo, "\n", sep="")
