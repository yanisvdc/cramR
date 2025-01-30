retrieve_and_validate_ML_model <- function(model_type, model_params, data, formula, custom_fit, custom_predict) {
  if (!is.null(model_type)) {
    # Retrieve model and validate user-specified parameters
    if (!is.null(learner_type) && learner_type == "fnn") {
      model_params <- validate_params_fnn(model_type, learner_type, model_params, X)
      model <- set_model(model_type, learner_type, model_params)
    } else {
      model <- set_model(model_type, learner_type, model_params)
      model_params <- validate_params(model, model_type, learner_type, model_params)
    }
  } else {
    # Custom mode: ensure custom_fit and custom_predict are specified
    if (is.null(custom_fit) || is.null(custom_predict)) {
      stop("As model_type is NULL (custom mode), custom_fit and custom_predict must be specified")
    }
    model <- NULL  # No predefined model in custom mode
  }
  return(list(model = model, model_params = model_params))
}
