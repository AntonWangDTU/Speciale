#################
# FUnction that cleanup temporary files building up when working with BPCells

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