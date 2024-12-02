#' Validate Parameters for Feedforward Neural Networks (FNNs)
#'
#' This function validates user-provided parameters for a Feedforward Neural Network (FNN) model.
#' It ensures the correct structure for \code{input_layer}, \code{layers}, \code{output_layer}, and \code{compile_args}.
#'
#' @param model_params A named list of parameters provided by the user for configuring the FNN model.
#' @return A named list of validated parameters merged with defaults for any missing values.
#' @export
validate_params_fnn <- function(model_params) {
  # Ensure model_params is a list
  if (!is.list(model_params)) {
    stop("`model_params` must be a list.")
  }

  # Ensure model_params contains required top-level keys
  required_keys <- c("input_layer", "layers", "output_layer", "compile_args")
  missing_keys <- setdiff(required_keys, names(model_params))
  if (length(missing_keys) > 0) {
    stop(paste("`model_params` must include the following top-level keys:", paste(missing_keys, collapse = ", ")))
  }

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
