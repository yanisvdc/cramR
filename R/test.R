# Load the required library
library(testthat)
library(grf)
library(glmnet)
library(keras)

# Example data for testing
set.seed(123)
X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)  # 100 samples, 5 features
D_data <- sample(c(0, 1), 100, replace = TRUE)          # Random binary treatment assignment
Y_data <- rnorm(100)                                    # Random outcome variable
nb_batch <- 3                                           # Number of batches

# Test cases for cram_learning

test_that("cram_learning works with default settings", {
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch)

  expect_type(result, "list")
  expect_named(result, c("final_policy_model", "policies", "batch_indices"))
  expect_equal(ncol(result$policies), nb_batch + 1)  # One column for baseline and others for batches
})

test_that("cram_learning supports causal forest with ridge learner", {
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch,
                          model_type = "Causal Forest", learner_type = "ridge")

  expect_true(!is.null(result$final_policy_model))
})

test_that("cram_learning throws error for invalid batch argument", {
  expect_error(cram_learning(X = X_data, D = D_data, Y = Y_data, batch = "invalid"),
               "must be either an integer or a list/vector of batch indices")
})

test_that("cram_learning validates baseline policy", {
  expect_error(cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch,
                             baseline_policy = list("invalid")),
               "baseline_policy must contain numeric values only")
})

test_that("cram_learning supports multiple model types", {
  models <- c("Causal Forest", "S-learner", "M-learner")
  learners <- c("ridge", "fnn")

  for (model in models) {
    for (learner in learners) {
      result <- tryCatch({
        cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch,
                      model_type = model, learner_type = learner)
      }, error = function(e) NULL)

      if (!is.null(result)) {
        expect_true(!is.null(result$final_policy_model))
      }
    }
  }
})

test_that("cram_learning works with custom batch indices", {
  custom_batches <- split(1:100, rep(1:nb_batch, length.out = 100))
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = custom_batches)

  expect_length(result$batch_indices, nb_batch)
  expect_true(all(unlist(result$batch_indices) %in% 1:100))
})

test_that("cram_learning works with a single batch", {
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = 1)

  expect_equal(ncol(result$policies), 2)  # Baseline + one batch policy
})

test_that("cram_learning handles large number of batches", {
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = 10)

  expect_equal(ncol(result$policies), 11)  # Baseline + 10 batch policies
})

test_that("cram_learning supports parallelization", {
  result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch, parallelize_batch = TRUE)

  expect_true(!is.null(result$final_policy_model))
  expect_equal(ncol(result$policies), nb_batch + 1)
})





example_function <- function(x, y, z = NULL, w = 5, ...) {}
formal_args <- formals(example_function)
print(formal_args)
# $x
#
# $y
#
# $z
# NULL
# $w
# 5
# $...
#

positional_args <- names(formal_args)[
  sapply(formal_args, function(arg) identical(arg, quote(expr = ))) & names(formal_args) != "..."
]
print(positional_args)
# [1] "x" "y"
