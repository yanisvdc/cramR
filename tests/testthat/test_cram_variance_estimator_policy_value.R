library(testthat)

test_that("cram_variance_estimator_policy_value returns a valid numeric result", {
  set.seed(123)
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- rnorm(100)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_variance_estimator_policy_value(X, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_variance_estimator_policy_value errors with mismatched input lengths", {
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- rnorm(99)  # length mismatch
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  expect_error(cram_variance_estimator_policy_value(X, Y, D, pi, batch_indices),
               "Y, D, and pi must have matching lengths")
})

test_that("cram_variance_estimator_policy_value handles constant Y and D values", {
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- rep(1, 100)
  D <- rep(1, 100)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_variance_estimator_policy_value(X, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_true(is.finite(result))
})

test_that("cram_variance_estimator_policy_value supports custom propensity function", {
  X <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
  Y <- rnorm(100)
  D <- sample(0:1, 100, replace = TRUE)
  pi <- matrix(runif(100 * 6), nrow = 100, ncol = 6)
  batch_indices <- split(1:100, rep(1:5, each = 20))
  custom_prop <- function(X) rep(0.6, nrow(X))

  result <- cram_variance_estimator_policy_value(X, Y, D, pi, batch_indices, propensity = custom_prop)

  expect_type(result, "double")
  expect_true(is.finite(result))
})
