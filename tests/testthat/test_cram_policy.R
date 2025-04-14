library(testthat)

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
  X <- matrix(rnorm(80 * 2), nrow = 80)
  D <- sample(0:1, 80, replace = TRUE)
  Y <- rnorm(80)
  batch <- rep(1:4, each = 20)

  res <- cram_policy(
    X, D, Y, batch,
    model_type = "s_learner",
    learner_type = "ridge"
  )

  expect_type(res, "list")
  expect_s3_class(res$raw_results, "data.frame")
})

test_that("cram_policy allows custom propensity function", {
  set.seed(99)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  propensity_fn <- function(X) rep(0.6, nrow(X))

  res <- cram_policy(
    X, D, Y, batch,
    propensity = propensity_fn
  )

  expect_true("Delta Estimate" %in% res$raw_results$Metric)
})

test_that("cram_policy returns correct structure for custom model function", {
  set.seed(1)
  X <- matrix(rnorm(100 * 2), nrow = 100)
  D <- sample(0:1, 100, replace = TRUE)
  Y <- rnorm(100)
  batch <- rep(1:5, each = 20)

  dummy_fit <- function(X, Y, D) lm(Y ~ D + ., data = data.frame(cbind(Y, D, X)))
  dummy_predict <- function(model, X_new, D_new) {
    cate <- predict(model, newdata = data.frame(newdata))

    # Apply decision rule: treat if CATE > 0
    return(as.integer(cate > 0))
  }

  res <- cram_policy(
    X, D, Y, batch,
    custom_fit = dummy_fit,
    custom_predict = dummy_predict
  )
  expect_true("Policy Value Estimate" %in% res$raw_results$Metric)
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
