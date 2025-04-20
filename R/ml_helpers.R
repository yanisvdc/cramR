get_loss_function <- function(loss_name) {
  loss_functions <- list(

    # Squared Error (SE) - Per individual (Regression)
    # y_pred: Numeric vector of predictions
    # y_true: Numeric vector of true values
    "se" = function(y_pred, y_true) {
      (y_pred - y_true) ^ 2
    },

    # Absolute Error (AE) - Per individual (Regression)
    "ae" = function(y_pred, y_true) {
      abs(y_pred - y_true)
    },

    # Log Loss - Per individual (Classification)
    # y_pred: Data frame with probabilities assigned to each label
    # y_true: Factor
    "logloss" = logloss <- function(y_pred, y_true) {
      if (!is.factor(y_true)) {
        stop("y_true must be a factor with levels matching the column names of y_pred.")
      }

      # Ensure predicted probabilities are numeric matrix
      y_pred <- as.matrix(y_pred)

      # Safety: clamp predicted probabilities to avoid log(0)
      eps <- 1e-15
      y_pred <- pmax(pmin(y_pred, 1 - eps), eps)

      # Convert factor to character to match column names
      class_labels <- as.character(y_true)

      # Get probability assigned to the true class for each observation
      row_indices <- seq_len(nrow(y_pred))
      col_indices <- match(class_labels, colnames(y_pred))

      if (any(is.na(col_indices))) {
        stop("Some labels in y_true do not match column names of y_pred.")
      }

      probs_for_true <- y_pred[cbind(row_indices, col_indices)]

      # Return mean log loss
      -log(probs_for_true)
    },

    # Accuracy - Per individual (Classification)
    # y_pred: Factor
    # y_true: Factor
    "accuracy" = function(y_pred, y_true) {
      # Convert both factors to their labels (character) before comparing
      # Return vector with 1 if match and 0 otherwise
      as.numeric(as.character(y_pred) == as.character(y_true))
    }
  )

  if (!(loss_name %in% names(loss_functions))) {
    stop("Error: Loss function not recognized. Choose from: ", paste(names(loss_functions), collapse = ", "))
  }

  return(loss_functions[[loss_name]])
}


compute_loss <- function(ml_preds, data, formula=NULL, loss_name) {

  # Get the appropriate loss function
  loss_function <- get_loss_function(loss_name)

  # Extract the response variable from the formula
  target_var <- all.vars(formula)[1]
  true_y <- data[[target_var]]  # Actual target values

  if (loss_name %in% c("logloss", "accuracy")) {
    unique_vals <- sort(unique(true_y))
    labels <- paste0("class", unique_vals)
    true_y <- factor(true_y, levels = unique_vals, labels = labels)
  }

  if (is.data.frame(ml_preds)) {
    num_obs <- nrow(ml_preds)
  } else {
    num_obs <- length(ml_preds)
  }

  # Ensure ml_pred and true_y have the same length
  if (num_obs != length(true_y)) {
    stop(sprintf(
      "Error: Predictions and true values must have the same length.\nPredictions: %d rows, True values: %d rows.",
      num_obs, length(true_y)
    ))
  }

  # Compute per-instance loss
  individual_losses <- loss_function(y_pred = ml_preds, y_true = true_y)

  return(individual_losses)
}

create_cumulative_data_ml <- function(data, batches, nb_batch) {
  # Step 3: Create a data.table for cumulative batches
  # Initialize an empty list to store cumulative data for each batch
  cumulative_data_list <- lapply(1:nb_batch, function(t) {
    # Combine indices for batches 1 through t
    cumulative_indices <- unlist(batches[1:t])

    # Subset X, D, Y using cumulative indices
    list(
      t = t,  # Add t as the index
      # cumulative_index = list(cumulative_indices),  # Optional: Store cumulative indices as a list in one row
      data_cumul = list(data[cumulative_indices, ])
    )
  })

  # Convert the list to a data.table
  cumulative_data_dt <- rbindlist(cumulative_data_list)

  # Explicitly return the cumulative data table
  return(cumulative_data_dt)
}


ensure_caret_dependencies <- function(method) {
  libs <- caret::getModelInfo(method, regex = FALSE)[[1]]$library
  missing <- libs[!vapply(libs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("The following packages are required for method = '", method,
         "': ", paste(missing, collapse = ", "),
         ". Please install them manually.")
  }
}

