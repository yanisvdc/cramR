#' Predict with the Specified Model
#'
#' This function performs inference using a trained model
#'
#' @param model A trained model object returned by the `fit_model_ml` function.
#' @param data The dataset
#' @param formula The formula
#' @param caret_params The parameters of the caret model
#' @return A vector of predictions or CATE estimates, depending on the \code{model_type} and \code{learner_type}.
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}
#' @importFrom grf causal_forest
#' @import glmnet
#' @import keras
#' @export
model_predict_ml <- function(model, data, formula, caret_params) {
  # Create a local copy to avoid modifying the user's data
  new_data <- data

  if (!is.null(formula)) {
    # Supervised Model: Remove response variable
    response_var <- all.vars(formula)[1]  # Extract response variable name
    if (response_var %in% colnames(new_data)) {
      new_data <- new_data[, !(colnames(new_data) %in% response_var), drop = FALSE]  # Remove response variable
    }

    # Generate predictions for supervised models
    predictions <- predict(model, newdata = new_data)

  } else if (!is.null(caret_params$method) && caret_params$method == "kmeans") {
    # If K-Means, return cluster assignments
    if (!"finalModel" %in% names(model) || !"cluster" %in% names(model$finalModel)) {
      stop("Error: K-Means model structure invalid. Ensure the model was trained correctly.")
    }
    predictions <- model$finalModel$cluster

  } else {
    # Error handling if model is not recognized
    stop("Error: Model not found or method not supported in caret_params.")
  }

  return(predictions)
}

