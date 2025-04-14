library(testthat)
library(DT)

test_that("cram_bandit works with 3D pi input", {
  set.seed(42)
  T <- 100
  K <- 4

  pi <- array(runif(T * T * K, 0.1, 1), dim = c(T, T, K))
  for (t in 1:T) {
    for (j in 1:T) {
      pi[j, t, ] <- pi[j, t, ] / sum(pi[j, t, ])
    }
  }

  reward <- rnorm(T, mean = 1, sd = 0.5)
  arm <- sample(1:K, T, replace = TRUE)

  result <- cram_bandit(pi, arm, reward)

  expect_type(result, "list")
  expect_named(result, c("raw_results", "interactive_table"))
  expect_s3_class(result$raw_results, "data.frame")
  expect_s3_class(result$interactive_table, "datatables")
  expect_equal(nrow(result$raw_results), 4)
  expect_true(all(c("Metric", "Value") %in% colnames(result$raw_results)))
})

test_that("cram_bandit works with 2D pi input", {
  set.seed(43)
  T <- 100
  pi <- matrix(runif(T * T, 0.1, 1), nrow = T, ncol = T)
  pi <- pi / rowSums(pi)  # row normalization for meaningful structure

  reward <- rnorm(T)
  arm <- sample(1:2, T, replace = TRUE)

  result <- cram_bandit(pi, arm, reward)

  expect_type(result, "list")
  expect_named(result, c("raw_results", "interactive_table"))
  expect_s3_class(result$raw_results, "data.frame")
  expect_s3_class(result$interactive_table, "datatables")
  expect_equal(nrow(result$raw_results), 4)
  expect_true(all(is.finite(result$raw_results$Value)))
})
