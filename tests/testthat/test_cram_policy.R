library(testthat)
library(grf)
library(data.table)
library(glmnet)
# library(keras)

test_that("cram_policy runs correctly with default causal_forest settings", {
  set.seed(123)
  N <- 100
  X <- matrix(rnorm(N * 3), nrow = N)
  D <- sample(0:1, N, replace = TRUE)
  Y <- rnorm(N)
  batch <- rep(1:5, length.out = N)

  result <- cram_policy(X, D, Y, batch)

  expect_type(result, "list")
  expect_named(result, c("raw_results", "interactive_table", "final_policy_model"))
  expect_s3_class(result$raw_results, "data.frame")
})

test_that("cram_policy works with ridge learner (s-learner)", {
  set.seed(42)
  X <- matrix(rnorm(200 * 2), nrow = 200)
  D <- sample(0:1, 200, replace = TRUE)
  Y <- rnorm(200)
  batch <- rep(1:5, each = 40)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "ridge"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with FNN user param (s-learner)", {

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

  set.seed(42)
  X <- matrix(rnorm(200 * 2), nrow = 200)
  D <- sample(0:1, 200, replace = TRUE)
  Y <- rnorm(200)
  batch <- rep(1:5, each = 40)

  input_shape <- ncol(X) + 1 # S-learner so add D

  default_model_params <- list(
    input_layer = list(units = 64, activation = 'relu', input_shape = input_shape),  # Define default input layer
    layers = list(
      list(units = 32, activation = 'relu')
    ),
    output_layer = list(units = 1, activation = 'linear'),
    compile_args = list(optimizer = 'adam', loss = 'mse'),
    fit_params = list(epochs = 5, batch_size = 32, verbose = 0)
  )

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "fnn",
    model_params = default_model_params
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with ridge learner (m-learner)", {
  set.seed(43)
  X <- matrix(rnorm(200 * 2), nrow = 200)
  D <- sample(0:1, 200, replace = TRUE)
  Y <- rnorm(200)
  batch <- rep(1:5, each = 40)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "m_learner",
    learner_type = "ridge"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with caret learner (m-learner)", {
  set.seed(43)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "m_learner",
    learner_type = "caret"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with caret learner (s-learner)", {
  set.seed(43)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "caret"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with caret learner classification (s-learner)", {
  set.seed(43)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- sample(c(0, 1), size = nrow(X), replace = TRUE)
  batch <- rep(1:5, each = 20)
  model_params <- list(formula = Y ~ ., caret_params = list(method = "rf", trControl = trainControl(method = "none", classProbs = TRUE)))

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "caret",
    model_params = model_params
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy works with fnn learner if keras is available", {

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

  set.seed(44)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "fnn"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy allows custom propensity function", {
  set.seed(99)
  X <- matrix(rnorm(200 * 2), nrow = 200)
  D <- sample(0:1, 200, replace = TRUE)
  Y <- rnorm(200)
  batch <- rep(1:5, each = 40)

  propensity_fn <- function(X) rep(0.6, nrow(X))

  res <- cram_policy(
    X, D, Y, batch,
    propensity = propensity_fn,
    model_type = "m_learner"
  )

  expect_true("Delta Estimate" %in% res$raw_results$Metric)
})

test_that("cram_policy returns correct structure for custom model function", {
  set.seed(1)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  colnames(X) <- paste0("X", seq_len(ncol(X)))  # Name columns X1, X2, ...

  dummy_fit <- function(X, Y, D) {
    df <- data.frame(Y = Y, D = D, X)
    lm(Y ~ D + ., data = df)
  }

  dummy_predict <- function(model, X_new, D_new) {
    newdata <- data.frame(D = D_new, X_new)
    predict(model, newdata = newdata) > 0
  }

  res <- cram_policy(
    X, D, Y, batch,
    model_type = NULL,
    custom_fit = dummy_fit,
    custom_predict = dummy_predict
  )
  expect_true("Policy Value Estimate" %in% res$raw_results$Metric)
})

test_that("cram_policy supports model_params input", {
  set.seed(12)
  X <- matrix(rnorm(120 * 3), nrow = 120)
  D <- sample(0:1, 120, replace = TRUE)
  Y <- rnorm(120)
  batch <- rep(1:6, each = 20)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "causal_forest",
    learner_type = NULL,
    model_params = list(num.trees = 10)
  )

  expect_true("Delta CI Upper" %in% res$raw_results$Metric)
})

test_that("cram_policy throws error on mismatched input lengths", {
  X <- matrix(rnorm(80 * 2), nrow = 80)
  D <- sample(0:1, 79, replace = TRUE)  # Mismatch
  Y <- rnorm(80)
  batch <- rep(1:4, each = 20)

  expect_error(
    cram_policy(X, D, Y, batch),
    "length"
  )
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


