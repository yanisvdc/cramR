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
  new_data <- data

  if (!is.null(formula)) {
    response_var <- all.vars(formula)[1]

    if (response_var %in% colnames(new_data)) {
      new_data <- new_data[, !(colnames(new_data) %in% response_var), drop = FALSE]
    }

    # If classification and probability prediction is enabled
    if (!is.null(caret_params$trControl) &&
        isTRUE(caret_params$trControl$classProbs)) {

      probs <- predict(model, newdata = new_data, type = "prob")

      # Get positive class name from model
      positive_class <- levels(model$trainingData$.outcome)[2]  # Typically "Yes"

      if (!(positive_class %in% colnames(probs))) {
        stop(sprintf("Error: Could not find predicted probability column for class '%s'.", positive_class))
      }

      predictions <- probs[, positive_class]

    } else {
      # Regular prediction (numeric or factor depending on model)
      predictions <- predict(model, newdata = new_data)
    }

  } else {
    stop("Error: Model not found or method not supported in caret_params.")
  }

  return(predictions)
}
