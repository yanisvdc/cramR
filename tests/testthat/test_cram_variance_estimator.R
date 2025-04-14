library(testthat)

test_that("cram_variance_estimator returns valid output with default propensity", {
  set.seed(42)
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- rnorm(100)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_variance_estimator(X, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_variance_estimator returns valid output with custom propensity function", {
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- rnorm(100)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))
  propensity_func <- function(X) rep(0.6, nrow(X))

  result <- cram_variance_estimator(X, Y, D, pi, batch_indices, propensity = propensity_func)

  expect_type(result, "double")
  expect_true(is.finite(result))
})

test_that("cram_variance_estimator errors on mismatched input sizes", {
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- rnorm(99)  # One short
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  expect_error(cram_variance_estimator(X, Y, D, pi, batch_indices),
               "Y, D, and pi must have matching lengths")
})

test_that("cram_variance_estimator handles all-zero Y and D vectors", {
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- rep(0, 100)
  D <- rep(0, 100)
  pi <- matrix(runif(100 * 6), nrow = 100)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_variance_estimator(X, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_true(is.finite(result))
})
