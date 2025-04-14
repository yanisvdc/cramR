library(testthat)

test_that("validate_params_fnn returns defaults if model_params is NULL", {
  X <- matrix(rnorm(20), ncol = 2)
  res <- validate_params_fnn("s_learner", "fnn", NULL, X)
  expect_true(is.list(res))
  expect_true(all(c("input_layer", "layers", "output_layer", "compile_args", "fit_params") %in% names(res)))
})

test_that("validate_params_fnn errors if input_layer is not a list", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_input <- list(
    input_layer = "notalist",
    layers = list(),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(
    validate_params_fnn("s_learner", "fnn", bad_input, X),
    "input_layer.*list"
  )
})

test_that("validate_params_fnn errors if input_layer missing fields", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_input <- list(
    input_layer = list(units = 32),  # missing activation and input_shape
    layers = list(),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_input, X), "input_layer.*include")
})

test_that("validate_params_fnn errors if layers are not list of lists", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_layers <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list("notalist"),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_layers, X), "layers.*list of lists")
})

test_that("validate_params_fnn errors if any layer is missing units or activation", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_layer <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list(list(units = 32)),  # missing activation
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_layer, X), "layers.*must include")
})

test_that("validate_params_fnn errors if output_layer is not valid", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_output <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list(list(units = 32, activation = "relu")),
    output_layer = "notalist",
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_output, X), "output_layer.*list")
})

test_that("validate_params_fnn errors if output_layer missing fields", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_output <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list(list(units = 32, activation = "relu")),
    output_layer = list(units = 1),  # missing activation
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_output, X), "output_layer.*include")
})

test_that("validate_params_fnn errors if compile_args are bad", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_compile <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list(list(units = 32, activation = "relu")),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam"),  # missing loss
    fit_params = list(epochs = 10, batch_size = 32)
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_compile, X), "compile_args.*loss")
})

test_that("validate_params_fnn errors if fit_params are bad", {
  X <- matrix(rnorm(20), ncol = 2)
  bad_fit <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = 3),
    layers = list(list(units = 32, activation = "relu")),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse"),
    fit_params = list(batch_size = 32)  # missing epochs
  )
  expect_error(validate_params_fnn("s_learner", "fnn", bad_fit, X), "fit_params.*epochs")
})
