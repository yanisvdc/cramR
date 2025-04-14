library(testthat)
library(data.table)

# Define the true DGPs consistent with your example
dgp_X <- function(n) {
  data.table(
    binary = rbinom(n, 1, 0.5),
    discrete = sample(1:5, n, replace = TRUE),
    continuous = rnorm(n)
  )
}

dgp_D <- function(X) {
  rbinom(nrow(X), 1, 0.5)
}

dgp_Y <- function(D, X) {
  theta <- ifelse(
    X[, binary] == 1 & X[, discrete] <= 2,
    1,
    ifelse(X[, binary] == 0 & X[, discrete] >= 4, -1, 0.1)
  )
  Y <- D * (theta + rnorm(length(D), 0, 1)) + (1 - D) * rnorm(length(D))
  return(Y)
}

test_that("cram_simulation runs correctly and returns expected metrics", {
  set.seed(1)

  # Define DGPs
  dgp_X <- function(n) data.table::data.table(
    binary = rbinom(n, 1, 0.5),
    discrete = sample(1:5, n, replace = TRUE),
    continuous = rnorm(n)
  )

  dgp_D <- function(X) rbinom(nrow(X), 1, 0.5)

  dgp_Y <- function(D, X) {
    theta <- ifelse(
      X[, binary] == 1 & X[, discrete] <= 2, 1,
      ifelse(X[, binary] == 0 & X[, discrete] >= 4, -1, 0.1)
    )
    Y <- D * (theta + rnorm(length(D))) + (1 - D) * rnorm(length(D))
    return(Y)
  }

  # Run the simulation
  result <- cram_simulation(
    dgp_X = dgp_X,
    dgp_D = dgp_D,
    dgp_Y = dgp_Y,
    batch = 20,
    nb_simulations = 2,
    nb_simulations_truth = 3,
    sample_size = 100
  )

  expect_type(result, "list")
  expect_named(result, c("raw_results", "interactive_table"))
  expect_s3_class(result$raw_results, "data.frame")
})
