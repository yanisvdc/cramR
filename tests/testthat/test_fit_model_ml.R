library(testthat)

test_that("fit_model_ml errors when formula is NULL", {
  dummy_data <- data.frame(x = rnorm(10), y = rnorm(10))
  caret_params <- list(method = "lm")

  expect_error(
    fit_model_ml(data = dummy_data, formula = NULL, caret_params = caret_params, classify = FALSE),
    "A formula must be provided"
  )
})

test_that("fit_model_ml errors when method is missing", {
  dummy_data <- data.frame(x = rnorm(10), y = rnorm(10))
  caret_params <- list()  # method missing

  expect_error(
    fit_model_ml(data = dummy_data, formula = y ~ x, caret_params = caret_params, classify = FALSE),
    "'method' must be specified"
  )
})

test_that("fit_model_ml sets default trControl if not provided", {
  dummy_data <- data.frame(x = rnorm(10), y = rnorm(10))
  caret_params <- list(method = "lm")  # no trControl

  model <- fit_model_ml(data = dummy_data, formula = y ~ x, caret_params = caret_params, classify = FALSE)

  expect_s3_class(model, "train")
  expect_true(is.list(model$control))
  expect_equal(model$control$method, "none")
})

