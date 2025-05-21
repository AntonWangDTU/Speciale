# Functions for cell typing
# Custom GPT cell type annotation function
predict_celltypes <- function(markers_df,
                              tissuename = "human tissue",
                              model = "openai_gpt4_turbo_128k",
                              top_n_genes = 10,
                              api_key = Sys.getenv("OPENAI_API_KEY"),
                              base_url = "https://api.marketplace.novo-genai.com/v1/chat/completions",
                              max_retries = 2) {
  
  # Add cluster column if missing (assume cluster 0 if only one)
  if (!"cluster" %in% colnames(markers_df)) {
    markers_df$cluster <- 0
    markers_df$gene <- rownames(markers_df)
  }
  
  # Prepare marker list
  marker_list <- markers_df %>%
    group_by(cluster) %>%
    arrange(p_val_adj, .by_group = TRUE) %>% # USE EITHER p_val_adj OR avg_log2FC for sorting 
    slice_head(n = top_n_genes) %>%
    summarise(genes = paste(gene, collapse = ", "), .groups = "drop") 
  
  # Save clusters to match results
  clusters <- marker_list$cluster
  
  build_prompt <- function(marker_list) {
    paste0(
      "Identify cell types of ", tissuename, " cells", 
      " using the following markers. Identify one cell type for each row. Only provide the cell type name.",
      " If you cannot identify a cell population given the markers, return 'Can't Identify'.",
      " Thus the output should return the same number of rows as the input. \n\n",
      paste(marker_list$genes, collapse = "\n")
    )
  }
  
  parse_response <- function(text) {
    lines <- unlist(strsplit(text, "\n"))
    lines <- trimws(lines)
    lines <- lines[lines != ""]
    gsub("^(\\d+[\\.\\-]\\s*|\\-\\s*)", "", lines)
  }
  
  # Retry logic
  for (attempt in 1:max_retries) {
    prompt <- build_prompt(marker_list)
    body <- list(
      model = model,
      messages = list(
        list(role = "user", content = prompt)
      )
    )
    
    res <- POST(
      url = base_url,
      add_headers(
        Authorization = paste("Bearer", api_key),
        `Content-Type` = "application/json"
      ),
      body = toJSON(body, auto_unbox = TRUE)
    )
    
    if (http_error(res)) {
      stop("API request failed: ", content(res, "text", encoding = "UTF-8"))
    }
    
    parsed <- content(res, "parsed", simplifyVector = TRUE)
    raw_output <- parsed[["choices"]]$message$content
    celltypes <- parse_response(raw_output)
    
    if (length(celltypes) == length(clusters)) {
      break
    } else {
      warning(sprintf(
        "Mismatch in expected (%d) and returned (%d) cell types on attempt %d.",
        length(clusters), length(celltypes), attempt
      ))
      print(raw_output)
    }
  }
  
  # Final padding if still incorrect
  if (length(celltypes) < length(clusters)) {
    warning(sprintf(
      "Mismatch in expected (%d) and returned (%d) cell types on attempt %d. Padding the missing number of celltypes.",
      length(clusters), length(celltypes), attempt
    ))
    celltypes <- c(celltypes, rep(NA, length(clusters) - length(celltypes)))
  }
  
  # Combine with clusters
  data.frame(
    cluster = clusters,
    actual_cell_type = sapply(clusters, function(c) {
      parts <- strsplit(c, "_")[[1]]
      if (length(parts) >= 4) parts[4] else NA
    }),
    predicted_cell_type = celltypes,
    stringsAsFactors = FALSE
  )
}

###############################################################################

evaluate_prediction_llm <- function(annotated_df, model = "openai_gpt4o_128k",
                                    api_key = Sys.getenv("OPENAI_API_KEY"),
                                    base_url = "https://api.marketplace.novo-genai.com/v1/chat/completions") {
  
  # Build prompt for LLM
  prompt <- paste0(
    "You are a biomedical cell type expert.\n",
    "Evaluate whether the predicted cell types match the true annotated cell types. ",
    "Cell types may be described at different levels of specificity or using different terminologies.\n\n",
    "Classify the match between each prediction and annotation into one of three categories:\n",
    "- 'Match': The predicted and actual cell types refer to the same concept, even if phrased differently.\n",
    "- 'Partial Match': The prediction is a subgroup, supergroup, or related state/type of the annotated cell type.\n",
    "- 'No Match': The predicted cell type does not reasonably correspond to the annotated one.\n\n",
    "Return a Markdown table with columns: cluster, actual_cell_type, predicted_cell_type, match (Match / Partial Match / No Match), and explanation (max 1 sentence).\n\n",
    "Here is the data:\n\n",
    paste0(
      apply(annotated_df, 1, function(row) {
        paste( "| Annotated:", row["actual_cell_type"], 
              "| Predicted:", row["predicted_cell_type"])
      }),
      collapse = "\n"
    )
  )
  
  # Prepare request body
  body <- list(
    model = model,
    messages = list(list(role = "user", content = prompt))
  )
  
  # Send request
  res <- POST(
    url = base_url,
    add_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ),
    body = toJSON(body, auto_unbox = TRUE)
  )
  
  # Error handling
  if (http_error(res)) {
    stop("API request failed: ", content(res, "text", encoding = "UTF-8"))
  }
  
  # Parse response
  parsed <- content(res, "parsed", simplifyVector = TRUE)
  output <- parsed[["choices"]]$message$content
  
  # Extract table lines
  lines <- unlist(strsplit(output, "\n"))
  lines <- lines[grepl("^\\|", lines)]  # Keep table rows
  lines <- lines[-c(1, 2)]  # Remove header + separator
  
  # Extract fields from each line
  parsed <- do.call(rbind, lapply(lines, function(line) {
    fields <- strsplit(gsub("^\\||\\|$", "", line), "\\|")[[1]]
    trimws(fields)
  }))
  
  df <- as.data.frame(parsed, stringsAsFactors = FALSE)
  colnames(df) <- c("Cluster", "actual_cell_type", "predicted_cell_type", "match", "explanation")
  
  # Normalize match values
  df$match <- tolower(trimws(df$match))
  df$match <- gsub("^(match|exact match)$", "Match", df$match, ignore.case = TRUE)
  df$match <- gsub("^partial.*", "Partial Match", df$match, ignore.case = TRUE)
  df$match <- gsub("^(no match|different.*|prediction.*unclear|unknown|cannot.*|can't.*)", "No Match", df$match, ignore.case = TRUE)
  df$match <- factor(df$match, levels = c("Match", "Partial Match", "No Match"))
  
  # Parse Cluster into components
  cluster_parts <- do.call(rbind, strsplit(annotated_df$cluster, "_"))
  df$Condition <- cluster_parts[, 1]
  df$Group <- cluster_parts[, 2]
  df$Organ <- cluster_parts[, 3] 
  df$Celltype <- cluster_parts[, 4]
  df$Query <- cluster_parts[,5]
  
  # Reorder columns
  df <- df[, c("Cluster", "Condition", "Group", "Organ", "Query", "Celltype", "predicted_cell_type", "match", "explanation")]
  
  # Remove Cluster column if not needed
  df$Cluster <- NULL
  
  return(df)
}

###############################################################################
compute_match_accuracy <- function(eval_df) {
  # Ensure 'match' is character or factor
  match_col <- tolower(as.character(eval_df$match))
  
  total <- length(match_col)
  match_only <- sum(match_col == "match")
  match_plus_partial <- sum(match_col %in% c("match", "partial match"))
  
  cat("LLM Evaluation Accuracy:\n")
  cat("- Exact Match Accuracy:", round(match_only / total, 3), "\n")
  cat("- Match + Partial Match Accuracy:", round(match_plus_partial / total, 3), "\n")
  
  # Return invisibly in case user wants to extract
  invisible(list(
    exact_match_accuracy = match_only / total,
    match_plus_partial_accuracy = match_plus_partial / total
  ))
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

run_prediction_for_celltype <- function(data_all, data_group, data_organ, group, organ, cell_type) {
  
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
                  only.pos = FALSE)
    } else {
      message(paste("RUNNING", label, "INCLUDING QUERY CELLS"))
      FindMarkers(data_obj,
                  ident.1 = ident_1,
                  min.pct = 0.1,
                  logfc.threshold = 0.25,
                  only.pos = FALSE)
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
  
  # Save top N markers per cluster
  marker_outfile <- file.path(MARKER_LOG_DIR, paste0(group, "_", organ, "_", cell_type, "_markers.csv"))
  top_markers <- markers %>%
    group_by(cluster) %>%
    top_n(TOP_N_GENES, avg_log2FC)
  write_csv(top_markers, marker_outfile)
  
  predict_celltypes(markers, top_n_genes = TOP_N_GENES, tissuename = organ, model = MODEL)
}

###############################################################################

cleanup_tmp_files <- function(){
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
        message("✅ Removed: ", path)
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