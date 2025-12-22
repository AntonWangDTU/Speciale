library(BPCells)
library(Seurat)
library(SeuratObject)
library(hdf5r)
library(dplyr)

###############################################################################
# Function that loads atlas from disco and down samples it.
# Input: Filename of .h5ad format
# Output: downsampled seurat object in .rds format

sample_h5ad_to_rds <- function(filename, samples = 20000, output_rds = NULL, seed = 123) {
  
  if (is.null(output_rds)) {
    output_rds <- paste0("disco_data/subsets/",
                         tools::file_path_sans_ext(basename(filename)), 
                         "_subset_", samples, ".rds")
  }
  
  # Ensure directory exists
  dir_path <- dirname(output_rds)
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    message("Created directory: ", dir_path)
  }
  
  organ <- sub("_.*|\\.h5ad$", "", basename(filename))
  set.seed(seed)
  
  message("Opening .h5ad matrix as BPCells object...")
  bp_mat <- open_matrix_anndata_hdf5(filename)
  
  message("Reading metadata from .h5ad...")
  h5 <- H5File$new(filename, mode = "r")
  
  obs_names <- h5[["obs"]]$ls()$name
  obs_list <- lapply(obs_names, function(nm) {
    h5[["obs"]][[nm]]$read()
  })
  
  obs_df <- as.data.frame(obs_list, stringsAsFactors = FALSE)
  colnames(obs_df) <- obs_names
  
  # Use _index as rownames
  cell_ids <- h5[["obs/_index"]]$read()
  rownames(obs_df) <- cell_ids
  
  h5$close_all()
  
  # Select / reorder metadata columns
  drop_cols <- c("_index","age","age_group", "gender","race",
                 "subject_id","sample_type","sample_id","project_id")
  keep_order <- c("nFeature_RNA", "nCount_RNA", "cell_type","tissue",
                  "anatomical_site", "cell_sorting", "disease",
                  "platform","rna_source")
  
  obs_df <- obs_df %>%
    select(-any_of(drop_cols)) %>%
    select(any_of(keep_order))
  
  # Sample cells
  total_cells <- ncol(bp_mat)
  if (samples > total_cells) {
    warning("samples exceeds total number of cells. Using all cells.")
    samples <- total_cells
  }
  
  message("Sampling ", samples, " cells from total ", total_cells, " cells...")
  chosen_idx <- sample(seq_len(total_cells), samples)
  
  # Subset BPCells matrix (columns = cells)
  message("Subsetting BPCells matrix to selected cells...")
  bp_mat_subset <- bp_mat[, chosen_idx]
  
  # Write subset matrix to disk
  subset_dir <- file.path(dir_path, paste0("bp_subset_", organ, "_", samples))
  if (!dir.exists(subset_dir)) dir.create(subset_dir, recursive = TRUE)
  
  message("Writing subsetted BPCells matrix to: ", subset_dir)
  write_matrix_dir(bp_mat_subset, dir = subset_dir, overwrite = TRUE)
  
  # Reload subsetted matrix from disk (clean reference)
  counts_mat <- open_matrix_dir(subset_dir)
  
  # Build Seurat object
  message("Creating Seurat object...")
  seurat_obj <- CreateSeuratObject(
    counts = counts_mat,
    meta.data = obs_df[chosen_idx, , drop = FALSE]
  )
  
  # Save RDS
  message("Saving Seurat object to ", output_rds, " ...")
  saveRDS(seurat_obj, output_rds)
  
  message("Done! Seurat object created and saved.")
}
