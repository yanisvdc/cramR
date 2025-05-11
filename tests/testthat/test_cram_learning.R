library(testthat)
library(waldo)

# Test suite for cram_learning

test_that("cram_learning works with valid inputs and causal_forest", {
  set.seed(123)
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  D <- sample(c(0, 1), 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- 5

  result <- cram_learning(X = X, D = D, Y = Y, batch = batch, model_type = "causal_forest")

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
  expect_equal(ncol(result$policies), batch + 1)  # Baseline + batch policies
  expect_true(is.list(result$final_policy_model))
})

test_that("cram_learning works with ridge regression (s_learner)", {
  set.seed(123)
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  D <- sample(c(0, 1), 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- 3

  result <- cram_learning(X = X, D = D, Y = Y,
                          batch = batch,
                          model_type = "s_learner",
                          learner_type = "ridge",
                          model_params = list(nfolds = 3))

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
  expect_equal(ncol(result$policies), batch + 1)
  expect_true(is.list(result$final_policy_model))
})

test_that("cram_learning works with fnn (m_learner)", {

  skip("Skipping this test temporarily")

  testthat::skip_on_cran()

  # Check if keras is installed AND properly configured
  if (!requireNamespace("keras", quietly = TRUE)) {
    testthat::skip("keras not installed")
  }

  # Check if Keras backend is actually working
  if (!keras::is_keras_available()) {
    testthat::skip("Keras backend not available")
  }

  set.seed(123)
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  D <- sample(c(0, 1), 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- 5

  result <- cram_learning(X = X, D = D, Y = Y, batch = batch, model_type = "m_learner", learner_type = "fnn")

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
  expect_equal(ncol(result$policies), batch + 1)
})

test_that("cram_learning throws error for mismatched lengths", {
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  D <- sample(c(0, 1), 100, replace = TRUE)
  Y <- rnorm(99)  # Incorrect length
  batch <- 5

  # Test will pass if any error is raised
  expect_error(cram_learning(X = X, D = D, Y = Y, batch = batch))
})


test_that("cram_learning handles custom_fit and custom_predict", {
  set.seed(123)
  X <- matrix(rnorm(80 * 3), nrow = 80, ncol = 3)
  D <- sample(c(0, 1), 80, replace = TRUE)
  Y <- rnorm(80)
  batch <- 5

  custom_fit <- function(X, Y, D) {
    # Combine X and D into a single matrix
    X_combined <- cbind(X, D)

    # Fit a simple linear regression model
    model <- lm(Y ~ ., data = as.data.frame(X_combined))

    return(model)
  }

  # Custom S-Learner Predict
  custom_predict <- function(model, X_new, D_new) {
    # Create data frames for treated (D = 1) and untreated (D = 0)
    X_treated <- as.data.frame(cbind(X_new, D = 1))
    X_control <- as.data.frame(cbind(X_new, D = 0))

    # Predict potential outcomes
    Y_treated_pred <- predict(model, newdata = X_treated)
    Y_control_pred <- predict(model, newdata = X_control)

    # Calculate CATE as the difference between treated and control predictions
    cate <- Y_treated_pred - Y_control_pred

    return(as.numeric(cate)) # Return as a numeric vector
  }

  result <- cram_learning(X = X, D = D, Y = Y, batch = batch,
                          model_type = NULL, custom_fit = custom_fit, custom_predict = custom_predict)

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
})

test_that("cram_learning works with provided batch assignments", {
  set.seed(123)
  X <- matrix(rnorm(100 * 3), nrow = 100, ncol = 3)
  D <- sample(c(0, 1), 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- sample(1:3, 100, replace = TRUE)

  result <- cram_learning(X = X, D = D, Y = Y,
                          batch = batch,
                          model_type = "s_learner",
                          learner_type = "ridge",
                          model_params = list(nfolds = 3))

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
  expect_true(length(result$batch_indices) == length(unique(batch)))
})

test_that("cleanup __pycache__", {
  if (dir.exists("__pycache__")) {
    unlink("__pycache__", recursive = TRUE)
  }
  expect_false(dir.exists("__pycache__"))
})

# testthat::teardown({
#   tmp_dir <- tempdir()
#   autograph_files <- list.files(tmp_dir, pattern = "^__autograph_generated_file.*\\.py$", full.names = TRUE)
#   if (length(autograph_files) > 0) unlink(autograph_files, force = TRUE)
# })

test_that("cleanup keras temp files", {
  withr::defer({
    tmp_dir <- tempdir()
    autograph_files <- list.files(tmp_dir, pattern = "^__autograph_generated_file.*\\.py$", full.names = TRUE)
    if (length(autograph_files) > 0) unlink(autograph_files, force = TRUE)
  }, teardown_env())
  expect_true(TRUE)  # dummy expectation to pass the test
})
