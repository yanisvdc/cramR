#' Fit Model ML
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
  fitted_model <- do.call(caret::train, c(list(formula, data = data), caret_params))

  return(fitted_model)
}




# # Load necessary library
# library(caret)
#
# # Set seed for reproducibility
# set.seed(42)
#
# # Generate example dataset
# X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
# Y_data <- rnorm(100)  # Continuous target variable for regression
#
# # Combine into a single data frame
# data_df <- data.frame(X_data, Y = Y_data)
#
# # Define train control settings to use 10-fold cross-validation
# ctrl <- trainControl(method = "cv", number = 10)
#
# # Train the model using cross-validation
# model <- train(
#   Y ~ .,  # Formula specifying the model
#   data = data_df,
#   method = "lm",  # Linear regression method
#   trControl = ctrl  # Use the cross-validation control
# )
#
# # Print the model summary
# print(model)


# # Load necessary library
# library(caret)
#
# # Set seed for reproducibility
# set.seed(42)
#
# # Generate example dataset
# X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
# Y_data <- rnorm(100)  # Continuous target variable for regression
#
# # Combine into a single data frame
# data_df <- data.frame(X_data, Y = Y_data)
#
# # Define caret parameters for simple linear regression with NO resampling
# train_control <- trainControl(method = "none")  # No resampling, fit once on full data
#
# # Train the model using caret (formula-based approach)
# model <- caret::train(
#   Y ~ .,  # Linear regression formula
#   data = data_df,
#   method = "lm",  # Linear regression model
#   trControl = train_control  # No resampling
# )
#
# # Print the model summary
# print(model)
