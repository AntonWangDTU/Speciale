# Load libraries
library(tidyverse)
library(httr)
library(jsonlite)

###############################################################################

predict_celltypes <- function(markers_df,
                              tissuename = "human tissue",
                              model = "openai_gpt4_turbo_128k",
                              top_n_genes = 10,
                              api_key = Sys.getenv("OPENAI_API_KEY"),
                              base_url = "https://api.marketplace.novo-genai.com/v1/chat/completions",
                              max_retries = 2) {
  
  clusters <- markers_df$Celltype
  
  build_prompt <- function(markers_df) {
    paste0(
      "Identify cell types of ", tissuename, " cells", 
      " using the following markers. Identify one cell type for each row. Only provide the cell type name.",
      " If you cannot identify a cell population given the markers, return 'Can't Identify'.",
      " Thus the output should return the same number of rows as the input. \n\n",
      paste(markers_df$genes, collapse = "\n")
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
    prompt <- build_prompt(markers_df)
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
    
    # Parse the output
    parsed <- content(res, "parsed", simplifyVector = TRUE)
    raw_output <- parsed[["choices"]]$message$content
    celltypes <- parse_response(raw_output)
    
    # Check for mismatch in number of output cells
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
  
  # Create data frame
  markers_df$predicted_cell_type <- celltypes
  
  return(markers_df)
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
        paste( "| Annotated:", row["Celltype"], 
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
  annotated_df <- annotated_df %>% 
    mutate(
      match = df$match,
      explanation = df$explanation
    )
  
  # Reorder columns
  df <- annotated_df[, c("Condition", "Group", "Organ", "Query", "Celltype", "predicted_cell_type", "match", "explanation", "genes")]
  
  
  return(df)
}

################################################################################

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