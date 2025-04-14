library(testthat)

test_that("cram_policy_value_estimator returns a valid numeric estimate", {
  set.seed(123)

  N <- 100
  nb_batch <- 5
  X <- matrix(rnorm(N * 5), nrow = N)
  Y <- sample(0:1, N, replace = TRUE)
  D <- sample(0:1, N, replace = TRUE)
  pi <- matrix(runif(N * (nb_batch + 1)), nrow = N)  # nb_batch + 1 columns
  batch_indices <- split(1:N, rep(1:nb_batch, each = N / nb_batch))

  result <- cram_policy_value_estimator(X, Y, D, pi, batch_indices)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_policy_value_estimator throws error for mismatched lengths", {
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:1, 99, replace = TRUE)  # Wrong length
  pi <- matrix(runif(100 * 6), nrow = 100)  # 5 batches + 1 column
  batch_indices <- split(1:100, rep(1:5, each = 20))

  expect_error(
    cram_policy_value_estimator(X, Y, D, pi, batch_indices),
    "Y, D, and pi must have matching lengths"
  )
})

test_that("cram_policy_value_estimator throws error if D is not binary", {
  X <- matrix(rnorm(100 * 5), nrow = 100)
  Y <- sample(0:1, 100, replace = TRUE)
  D <- sample(0:2, 100, replace = TRUE)  # Not binary
  pi <- matrix(runif(100 * 6), nrow = 100)
  batch_indices <- split(1:100, rep(1:5, each = 20))

  expect_error(
    cram_policy_value_estimator(X, Y, D, pi, batch_indices),
    "D must be a binary vector"
  )
})

test_that("cram_policy_value_estimator works with custom propensity function", {
  set.seed(42)
  N <- 80
  nb_batch <- 4
  X <- matrix(rnorm(N * 3), nrow = N)
  Y <- sample(0:1, N, replace = TRUE)
  D <- sample(0:1, N, replace = TRUE)
  pi <- matrix(runif(N * (nb_batch + 1)), nrow = N)
  batch_indices <- split(1:N, rep(1:nb_batch, each = N / nb_batch))

  propensity_fn <- function(X) rep(0.6, nrow(X))

  result <- cram_policy_value_estimator(X, Y, D, pi, batch_indices, propensity = propensity_fn)

  expect_type(result, "double")
  expect_true(is.finite(result))
})
