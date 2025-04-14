library(testthat)

test_that("cram_bandit_est works with 2D pi input", {
  set.seed(123)
  T <- 100
  pi <- matrix(runif(T * 10), nrow = T)
  reward <- rnorm(T)
  arm <- sample(1:2, T, replace = TRUE)

  result <- cram_bandit_est(pi, reward, arm, batch = 10)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_bandit_est works with 3D pi input", {
  set.seed(42)
  T <- 100
  K <- 4
  pi <- array(runif(T * 10 * K, 0.1, 1), dim = c(T, 10, K))

  reward <- rnorm(T, mean = 1, sd = 0.5)
  arm <- sample(1:K, T, replace = TRUE)

  result <- cram_bandit_est(pi, reward, arm, batch = 10)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})
