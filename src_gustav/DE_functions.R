run_DE <- function(data_all, data_group, data_organ, group, organ, cell_type) {
  
  target_ident <- paste(organ, cell_type, sep = "_")
  
  # Helper: run FindMarkers with optional exclusion of the same cell_type
  run_markers <- function(data_obj, ident_1, exclude_celltype = FALSE, label) {
    if (exclude_celltype) {
      message(paste("RUNNING", label, "EXCLUDING QUERY CELLS"))
      cells_excl <- data_obj@meta.data %>%
        dplyr::filter(Celltype != !!cell_type) %>%
        dplyr::pull(cell_id)
      message(paste("Cells excluded:", ncol(data_obj) - length(cells_excl)))
      
      FindMarkers(data_obj,
                  ident.1 = ident_1,
                  ident.2 = cells_excl,
                  min.pct = 0.1,
                  logfc.threshold = 0.25,
                  only.pos = TRUE)
    } else {
      message(paste("RUNNING", label, "INCLUDING QUERY CELLS"))
      FindMarkers(data_obj,
                  ident.1 = ident_1,
                  min.pct = 0.1,
                  logfc.threshold = 0.25,
                  only.pos = TRUE)
    }
  }
  
  # Run all 6 marker sets
  markers_all_query     <- run_markers(data_all,   target_ident, FALSE, "ALL_QUERY")
  markers_all_noquery   <- run_markers(data_all,   target_ident, TRUE,  "ALL_NOQUERY")
  markers_group_query   <- run_markers(data_group, target_ident, FALSE, "GROUP_QUERY")
  markers_group_noquery <- run_markers(data_group, target_ident, TRUE,  "GROUP_NOQUERY")
  markers_organ_query   <- run_markers(data_organ, target_ident, FALSE, "ORGAN_QUERY")
  markers_organ_noquery <- run_markers(data_organ, target_ident, TRUE,  "ORGAN_NOQUERY")
  
  # Combine and predict
  markers <- create_marker_df(
    markers_all_query, markers_all_noquery,
    markers_group_query, markers_group_noquery,
    markers_organ_query, markers_organ_noquery,
    group, organ, cell_type
  )
}

###############################################################################

create_marker_df <- function(
    markers_all_query, markers_all_noquery,
    markers_group_query, markers_group_noquery,
    markers_organ_query, markers_organ_noquery,
    group, organ, cell_type
) {
  
  markers_all_query$cluster <- paste("All", organ, group, cell_type, "query", sep = "_")
  markers_all_noquery$cluster <- paste("All", organ, group, cell_type, "noquery", sep = "_")
  
  markers_group_query$cluster <- paste("Group", organ, group, cell_type, "query", sep = "_")
  markers_group_noquery$cluster <- paste("Group", organ, group, cell_type, "noquery", sep = "_")
  
  markers_organ_query$cluster <- paste("Organ", organ, group, cell_type, "query", sep = "_")
  markers_organ_noquery$cluster <- paste("Organ", organ, group, cell_type, "noquery", sep = "_")
  
  
  markers_combined <- bind_rows(markers_all_query, markers_all_noquery, 
                                markers_group_query, markers_group_noquery,
                                markers_organ_query, markers_organ_noquery) 
  
  markers_combined$gene <- gsub("\\.\\.\\..*", "", rownames(markers_combined))
  
  return(markers_combined)
}

###############################################################################

cleanup_tmp_files <- function(Verbose = FALSE){
  # Define the temp directory (adjust if needed)
  temp_dir <- tempdir()
  
  # List all files and folders in the temp directory
  files <- list.files(temp_dir, full.names = TRUE, recursive = TRUE)
  
  # Function to try removing a file/folder safely
  safe_remove <- function(path) {
    tryCatch({
      if (file.exists(path)) {
        # If it's a directory, remove recursively
        if (dir.exists(path)) {
          unlink(path, recursive = TRUE, force = FALSE)
        } else {
          file.remove(path)
        }
        if (Verbose){
          message("✅ Removed: ", path)
        }
      }
    }, error = function(e) {
      message("⚠️ Skipped (error): ", path, " - ", e$message)
    }, warning = function(w) {
      message("⚠️ Skipped (warning): ", path, " - ", w$message)
    })
  }
  
  # Loop through and try to safely remove each file/folder
  invisible(lapply(files, safe_remove))
}