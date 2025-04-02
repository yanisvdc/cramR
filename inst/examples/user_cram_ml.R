# Install devtools if not already installed
install.packages("devtools")

# Install the cramR package from your GitHub repository
devtools::install_github("yanisvdc/cramR")

# Load the package
library(cramR)

library(data.table)
library(glmnet)


# CRAM ML - package models ----------------------------------------------------------------

# Load necessary libraries
library(caret)

# Set seed for reproducibility
set.seed(42)

# Generate example dataset
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_data <- rnorm(100)  # Continuous target variable for regression
data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

# Define caret parameters for simple linear regression (no cross-validation)
caret_params_lm <- list(
  method = "lm",
  trControl = trainControl(method = "none")
)

# Define the batch count (not used in this simple example)
nb_batch <- 5
# nb_batch <- rep(1:5, each = 20)


# Run ML learning function
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,  # Linear regression model
  batch = nb_batch,
  loss_name = 'mse',
  caret_params = caret_params_lm
)

print(result)

# NB: possible loss_name and caret_params: 
# full parameter list: https://topepo.github.io/caret/model-training-and-tuning.html#model-training-and-parameter-tuning

# loss_name can be: "mse" (Mean Squared Error), "rmse" (Root Mean Squared Error), 
# "mae" (Mean Absolute Error), "logloss" (Binary Log Loss), "accuracy" (Classification Accuracy),
# "euclidean_distance" (Squared Euclidean Distance for K-Means Clustering)

# caret_params can include:
# - method: Specifies the machine learning algorithm (e.g., "lm" for linear regression, 
#   "rf" for random forest, "xgbTree" for XGBoost, "svmLinear" for Support Vector Machines)
# - trControl: Defines the resampling method (e.g., trainControl(method = "cv", number = 5) for 5-fold CV, 
#   or trainControl(method = "none") for no resampling)
# - tuneGrid: A data frame specifying hyperparameters for tuning (e.g., expand.grid(mtry = c(2, 3, 4)) for Random Forest)
# - metric: Specifies the performance metric for model selection (e.g., "RMSE" for regression, "Accuracy" for classification)
# - preProcess: Preprocessing steps to apply (e.g., c("center", "scale") for normalization)
# - importance: Boolean flag for computing variable importance (default = FALSE, often TRUE for tree-based models)

# CRAM ML - custom fit, predict and loss ------------------------------------------------------

# Set seed for reproducibility
set.seed(42)

# Define custom fit function (train model)
custom_fit <- function(data) {
  # Manually define the formula
  model <- lm(Y ~ x1 + x2 + x3, data = data)
  return(model)
}

# Define custom predict function
custom_predict <- function(model, data) {
  predictors_only <- data[, setdiff(names(data), "Y"), drop = FALSE]  # Exclude target column
  predict(model, newdata = predictors_only)
}

# Define custom loss function (Mean Squared Error)
custom_loss <- function(predictions, data) {
  actuals <- data$Y
  mse_loss <- (predictions - actuals)^2
  return(mse_loss)
}

# Run ML learning function with custom model
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,  # Linear regression model
  batch = nb_batch,
  custom_fit = custom_fit,
  custom_predict = custom_predict,
  custom_loss = custom_loss
)

# Print results
print(result)