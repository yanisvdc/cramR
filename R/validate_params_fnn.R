#' Cram Policy: Validate Parameters for Feedforward Neural Networks (FNNs)
#'
#' This function validates user-provided parameters for a Feedforward Neural Network (FNN) model.
#' It ensures the correct structure for \code{input_layer}, \code{layers}, \code{output_layer},
#' \code{compile_args} and \code{fit_params}.
#'
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}. Note: you can also set model_type to NULL and specify custom_fit and custom_predict to use your custom model.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression, \code{"fnn"} for Feedforward Neural Network and \code{"caret"} for Caret. Default is \code{"ridge"}. if model_type is 'causal_forest', choose NULL, if model_type is 's_learner' or 'm_learner', choose between 'ridge', 'fnn' and 'caret'.
#' @param model_params A named list of parameters provided by the user for configuring the FNN model.
#' @param X A matrix or data frame of covariates for which the parameters are validated.
#' @return A named list of validated parameters merged with defaults for any missing values.
#' @export
validate_params_fnn <- function(model_type, learner_type, model_params, X) {
  if (is.null(model_params)){
    # Determine the input shape based on model_type
    input_shape <- if (model_type == "s_learner") ncol(X) + 1 else ncol(X)

    default_model_params <- list(
      input_layer = list(units = 64, activation = 'relu', input_shape = input_shape),  # Define default input layer
      layers = list(
        list(units = 32, activation = 'relu')
      ),
      output_layer = list(units = 1, activation = 'linear'),
      compile_args = list(optimizer = 'adam', loss = 'mse'),
      fit_params = list(epochs = 5, batch_size = 32, verbose = 0)
    )
    return(default_model_params)
  }
  # If the previous test did not return, the user specified model_params
  # Ensure model_params is a list
  if (!is.list(model_params)) {
    stop("`model_params` must be a list.")
  }

  # Ensure model_params contains required top-level keys
  required_keys <- c("input_layer", "layers", "output_layer", "compile_args", "fit_params")
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

  # Validate fit_params
  if (!is.list(model_params$fit_params)) {
    stop("`fit_params` must be a list specifying `epochs`, `batch_size`, and optionally `verbose`.")
  }
  if (!all(c("epochs", "batch_size") %in% names(model_params$fit_params))) {
    stop("`fit_params` must include `epochs` and `batch_size`.")
  }

  return(model_params)
}
