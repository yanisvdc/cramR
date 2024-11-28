#' Set the Model Based on Type and Learner
#'
#' This function maps the model type and learner type to the corresponding model function.
#'
#' @param model_type A string specifying the model type. Supported options are \code{"Causal Forest"}, \code{"S-learner"}, \code{"M-learner"}.
#' @param learner_type A string specifying the learner type. Supported options depend on \code{model_type}.
#' @return The R function corresponding to the specified model.
#' @examples
#' set_model("Causal Forest", NULL)
#' set_model("S-learner", "ridge")
#' @export
set_model <- function(model_type, learner_type) {
  if (model_type == "Causal Forest") {
    # For Causal Forest
    model <- grf::causal_forest
  } else if (model_type == "S-learner" && learner_type == "ridge") {
    # For S-learner with Ridge Regression
    model <- glmnet::cv.glmnet
  } else if (model_type == "S-learner" && learner_type == "FNN") {
    # For S-learner with Feedforward Neural Network
    model <- function(X, Y, ...) {
      keras_model_sequential() %>%
        layer_dense(units = 64, activation = 'relu', input_shape = ncol(X)) %>%
        layer_dense(units = 32, activation = 'relu') %>%
        layer_dense(units = 1) %>%
        compile(optimizer = 'adam', loss = 'mse')
    }
  } else if (model_type == "M-learner" && learner_type == "ridge") {
    # For M-learner with Ridge Regression
    model <- glmnet::cv.glmnet
  } else {
    stop("Unsupported model_type or learner_type.")
  }

  return(model)
}
