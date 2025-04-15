library(testthat)
library(caret)
library(cramR)

set.seed(42)

# Base dataset for regression
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_reg <- rnorm(100)
data_df <- data.frame(X_data, Y = Y_reg)

# Base dataset for classification
Y_class <- factor(ifelse(rbinom(100, 1, 0.5) == 1, "Yes", "No"))
data_df_class <- data.frame(X_data, Y = Y_class)

# -----------------------------
# Built-in caret: Regression SE
# -----------------------------
test_that("cram_ml runs with lm and SE loss", {
  caret_params_lm <- list(
    method = "lm",
    trControl = trainControl(method = "none")
  )

  result <- cram_ml(
    data = data_df,
    formula = Y ~ .,
    batch = 5,
    loss_name = "se",
    caret_params = caret_params_lm
  )

  expect_type(result, "list")
  expect_named(result, c("raw_results", "interactive_table", "final_ml_model"))
  expect_s3_class(result$raw_results, "data.frame")
  expect_true(all(c("Expected Loss Estimate", "Expected Loss Standard Error") %in% result$raw_results$Metric))
})

# -----------------------------
# Built-in caret: Classification Accuracy + Logloss
# -----------------------------
test_that("cram_ml runs with glm and classification loss (accuracy + logloss)", {
  caret_params_glm_acc <- list(
    method = "glm",
    family = "binomial",
    trControl = trainControl(method = "none")
  )

  caret_params_glm_logloss <- list(
    method = "glm",
    family = "binomial",
    trControl = trainControl(method = "none", classProbs = TRUE)
  )

  result_acc <- cram_ml(
    data = data_df_class,
    formula = Y ~ .,
    batch = 5,
    loss_name = "accuracy",
    caret_params = caret_params_glm_acc
  )

  result_logloss <- cram_ml(
    data = data_df_class,
    formula = Y ~ .,
    batch = 5,
    loss_name = "logloss",
    caret_params = caret_params_glm_logloss
  )

  expect_type(result_acc$raw_results$Value, "double")
  expect_type(result_logloss$raw_results$Value, "double")
})

test_that("cram_ml runs with caret classification logloss + classProb TRUE", {
  # Set seed for reproducibility
  set.seed(42)

  # Generate example dataset
  X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
  # Y_data <- rnorm(100)  # Continuous target variable for regression
  # Test Y binary:
  Y_data <- factor(sample(c("no", "yes"), size = nrow(X_data), replace = TRUE), levels = c("no", "yes"))
  data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

  caret_params_lm <- list(method = "rf", trControl = trainControl(method = "none", classProbs = TRUE))

  nb_batch <- 5
  # nb_batch <- rep(1:5, each = 20)

  # Run ML learning function
  result <- cram_ml(
    data = data_df,
    formula = Y ~ .,
    batch = nb_batch,
    loss_name = 'logloss',
    caret_params = caret_params_lm
  )

  expect_type(result$raw_results$Value, "double")
})

test_that("cram_ml runs with caret classification accuracy + classProb FALSE", {
  # Set seed for reproducibility
  set.seed(42)

  # Generate example dataset
  X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
  # Y_data <- rnorm(100)  # Continuous target variable for regression
  # Test Y binary:
  Y_data <- factor(sample(c("no", "yes"), size = nrow(X_data), replace = TRUE), levels = c("no", "yes"))
  data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

  caret_params_lm <- list(method = "rf", trControl = trainControl(method = "none"))

  nb_batch <- 5
  # nb_batch <- rep(1:5, each = 20)

  # Run ML learning function
  result <- cram_ml(
    data = data_df,
    formula = Y ~ .,
    batch = nb_batch,
    loss_name = 'accuracy',
    caret_params = caret_params_lm
  )

  expect_type(result$raw_results$Value, "double")
})

# -----------------------------
# Custom fit, predict, loss
# -----------------------------
test_that("cram_ml works with full custom model + loss", {
  custom_fit <- function(data) {
    lm(Y ~ x1 + x2 + x3, data = data)
  }

  custom_predict <- function(model, data) {
    predict(model, newdata = data)
  }

  custom_loss <- function(pred, data) {
    (pred - data$Y)^2  # SE loss
  }

  result <- cram_ml(
    data = data_df,
    formula = Y ~ .,
    batch = 5,
    custom_fit = custom_fit,
    custom_predict = custom_predict,
    custom_loss = custom_loss
  )

  expect_type(result$raw_results$Value, "double")
  expect_s3_class(result$final_ml_model, "lm")
})
