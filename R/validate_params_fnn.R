#' Validate Parameters for Feedforward Neural Networks (FNNs)
#'
#' This function validates user-provided parameters for a Feedforward Neural Network (FNN) model.
#' It ensures the correct structure for \code{input_layer}, \code{layers}, \code{output_layer}, and \code{compile_args}.
#'
#' @param user_params A named list of parameters provided by the user for configuring the FNN model.
#' @return A named list of validated parameters merged with defaults for any missing values.
#' @export
validate_params_fnn <- function(user_params) {
  # Default FNN parameters
  default_params <- list(
    input_layer = list(units = 64, activation = "relu", input_shape = NULL),
    layers = list(
      list(units = 32, activation = "relu")
    ),
    output_layer = list(units = 1, activation = "linear"),
    compile_args = list(optimizer = "adam", loss = "mse")
  )

  # Merge user parameters with defaults
  model_params <- modifyList(default_params, user_params)

  # Validate input_layer
  if (!is.list(model_params$input_layer)) {
    stop("`input_layer` must be a list specifying units, activation, and input_shape.")
  }
  if (!all(c("units", "activation", "input_shape") %in% names(model_params$input_layer))) {
    stop("`input_layer` must include `units`, `activation`, and `input_shape`.")
  }

  # Validate layers
  if (!is.list(model_params$layers) || !all(sapply(model_params$layers, is.list))) {
    stop("`layers` must be a list of lists, where each sublist specifies a layer with `units` and `activation`.")
  }
  for (layer in model_params$layers) {
    if (!all(c("units", "activation") %in% names(layer))) {
      stop("Each layer in `layers` must include `units` and `activation`.")
    }
  }

  # Validate output_layer
  if (!is.list(model_params$output_layer)) {
    stop("`output_layer` must be a list specifying units and activation.")
  }
  if (!all(c("units", "activation") %in% names(model_params$output_layer))) {
    stop("`output_layer` must include `units` and `activation`.")
  }

  # Validate compile_args
  if (!is.list(model_params$compile_args)) {
    stop("`compile_args` must be a list specifying `optimizer`, `loss`, and optionally `metrics`.")
  }
  if (!all(c("optimizer", "loss") %in% names(model_params$compile_args))) {
    stop("`compile_args` must include at least `optimizer` and `loss`.")
  }

  return(model_params)
}
