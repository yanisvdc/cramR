# Function to return the appropriate loss function based on a string
get_loss_function <- function(loss_name) {
  loss_functions <- list(
    "mse" = MLmetrics::MSE,
    "rmse" = MLmetrics::RMSE,
    "mae" = MLmetrics::MAE,
    "mape" = MLmetrics::MAPE,
    "smape" = MLmetrics::SMAPE,
    "rmsle" = MLmetrics::RMSLE,
    "msle" = MLmetrics::MSLE,
    "logloss" = MLmetrics::LogLoss,
    "multilogloss" = MLmetrics::MultiLogLoss,
    "f1_score" = MLmetrics::F1_Score,
    "accuracy" = MLmetrics::Accuracy,
    "auc" = MLmetrics::AUC,
    "precision" = MLmetrics::Precision,
    "recall" = MLmetrics::Recall
  )

  if (!(loss_name %in% names(loss_functions))) {
    stop("Error: Loss function not recognized. Choose from: ", paste(names(loss_functions), collapse = ", "))
  }

  return(loss_functions[[loss_name]])
}

# Load required package
library(MLmetrics)

# Function to return the appropriate loss function based on a string
get_loss_function <- function(loss_name) {
  loss_functions <- list(
    "mse" = MLmetrics::MSE,
    "rmse" = MLmetrics::RMSE,
    "mae" = MLmetrics::MAE,
    "mape" = MLmetrics::MAPE,
    "smape" = MLmetrics::SMAPE,
    "rmsle" = MLmetrics::RMSLE,
    "msle" = MLmetrics::MSLE,
    "logloss" = MLmetrics::LogLoss,
    "multilogloss" = MLmetrics::MultiLogLoss,
    "f1_score" = MLmetrics::F1_Score,
    "accuracy" = MLmetrics::Accuracy,
    "auc" = MLmetrics::AUC,
    "precision" = MLmetrics::Precision,
    "recall" = MLmetrics::Recall
  )

  if (!(loss_name %in% names(loss_functions))) {
    stop("Error: Loss function not recognized. Choose from: ", paste(names(loss_functions), collapse = ", "))
  }

  return(loss_functions[[loss_name]])
}

# # Example usage:
# loss_name <- "mse"  # User inputs loss name as a string
# loss_fn <- get_loss_function(loss_name)  # Fetch corresponding loss function
#
# # Compute loss
# y_true <- c(3, -0.5, 2, 7)
# y_pred <- c(2.5, 0.0, 2, 8)
#
# loss_value <- loss_fn(y_true, y_pred)
# print(loss_value)

