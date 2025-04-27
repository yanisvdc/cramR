#' Cram ML: Fit Model ML
#'
#' This function trains a given unfitted model with the provided data and parameters,
#' according to model type and learner type.
#'
#' @param data The dataset
#' @param formula The formula
#' @param caret_params The parameters for caret model
#' @param classify Indicate if this is a classification problem. Defaults to FALSE
#' @return The fitted model object.
#' @importFrom caret trainControl
#' @export
fit_model_ml <- function(data, formula, caret_params, classify) {

  # Ensure `formula` is provided
  if (is.null(formula)) {
    stop("Error: A formula must be provided for model training.")
  }

  # Ensure `method` is specified
  if (!"method" %in% names(caret_params)) {
    stop("Error: 'method' must be specified in caret_params.")
  }

  # Set default trControl if not provided
  if (!"trControl" %in% names(caret_params)) {
    caret_params$trControl <- caret::trainControl(method = "none")  # Default to no resampling
  }

  # CARET CLASSIFICATION - Convert target into factors
  if (isTRUE(classify)) {
    target_var <- all.vars(formula)[1]  # Extract target variable from formula
    Y <- data[[target_var]]

    if (!is.factor(Y)) {
      unique_vals <- sort(unique(Y))
      labels <- paste0("class", unique_vals)
      data[[target_var]] <- factor(Y, levels = unique_vals, labels = labels)
    }
  }

  # Call caret::train() with formula
  ensure_caret_dependencies(caret_params$method)
  fitted_model <- do.call(caret::train, c(list(formula, data = data), caret_params))

  return(fitted_model)
}
