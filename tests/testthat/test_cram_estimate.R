library(testthat)
library(waldo)

# Test suite for cram_estimator

test_that("cram_estimator works with correct inputs", {
  set.seed(123)
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
  nb_batch <- 10
  batch_indices <- split(1:100, rep(1:nb_batch, each = 10))

  result <- cram_estimator(X_data, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_false(is.na(result))
})



test_that("cram_estimator throws an error for mismatched lengths", {
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(sample(0:1, 101 * 11, replace = TRUE), nrow = 101, ncol = 11) # Incorrect length
  batch_indices <- split(1:100, rep(1:10, each = 10))

  expect_error(cram_estimator(X_data, Y, D, pi, batch_indices),
               "Y, D, and pi must have matching lengths")
})

test_that("cram_estimator handles all zeros in Y", {
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- rep(0, 100)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
  batch_indices <- split(1:100, rep(1:10, each = 10))

  result <- cram_estimator(X_data, Y, D, pi, batch_indices)

  expect_equal(result, 0)
})

test_that("cram_estimator handles all ones in D", {
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- rep(1, 100)
  pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
  batch_indices <- split(1:100, rep(1:10, each = 10))

  result <- cram_estimator(X_data, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_false(is.na(result))
})

test_that("cram_estimator works with a single individual in each batch", {
  X_data <- matrix(rnorm(10 * 5), nrow = 10, ncol = 5)
  Y <- sample(0:1, 10, replace = TRUE)
  D <- sample(0:1, 10, replace = TRUE)
  pi <- matrix(sample(0:1, 10 * 11, replace = TRUE), nrow = 10, ncol = 11)
  batch_indices <- split(1:10, 1:10) # One individual per batch

  result <- cram_estimator(X_data, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_false(is.na(result))
})

test_that("cram_estimator throws an error for non-binary D", {
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:2, 100, replace = TRUE) # Non-binary D
  pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
  batch_indices <- split(1:100, rep(1:10, each = 10))

  expect_error(cram_estimator(X_data, Y, D, pi, batch_indices))
})

test_that("cram_estimator handles edge case with all zeros in pi", {
  X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(0, nrow = 100, ncol = 11) # All zeros in pi
  batch_indices <- split(1:100, rep(1:10, each = 10))

  result <- cram_estimator(X_data, Y, D, pi, batch_indices)

  expect_equal(result, 0)
})

test_that("cleanup __pycache__", {
  if (dir.exists("__pycache__")) {
    unlink("__pycache__", recursive = TRUE)
  }
  expect_false(dir.exists("__pycache__"))
})

