library(testthat)

test_that("cram_bandit_var works correctly with 3D pi input", {
  set.seed(42)
  T <- 100
  K <- 4
  pi <- array(runif(T * T * K, 0.1, 1), dim = c(T, T, K))

  # Normalize probabilities for each context and time step
  for (t in 1:T) {
    for (j in 1:T) {
      pi[j, t, ] <- pi[j, t, ] / sum(pi[j, t, ])
    }
  }

  reward <- rnorm(T, mean = 1, sd = 0.5)
  arm <- sample(1:K, T, replace = TRUE)

  result <- cram_bandit_var(pi, reward, arm)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_bandit_var works correctly with 2D pi input", {
  set.seed(43)
  T <- 100
  pi <- matrix(runif(T * T, 0.1, 1), nrow = T)
  pi <- pi / rowSums(pi)

  reward <- rnorm(T)
  arm <- sample(1:2, T, replace = TRUE)

  result <- cram_bandit_var(pi, reward, arm)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})
