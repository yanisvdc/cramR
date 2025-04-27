#' Cram ML: Predict with the Specified Model
#'
#' This function performs inference using a trained model
#'
#' @param model A trained model object returned by the `fit_model_ml` function.
#' @param data The dataset
#' @param formula The formula
#' @param caret_params The parameters of the caret model
#' @param cram_policy_handle Internal use. Post-process predictions differently for cram policy use. Defaults to FALSE.
#' @return Predictions of the model on the data
#' @importFrom grf causal_forest
#' @import glmnet
#' @import keras
#' @export
model_predict_ml <- function(model, data, formula, caret_params, cram_policy_handle=FALSE) {
  new_data <- data

  if (!is.null(formula)) {
    response_var <- all.vars(formula)[1]

    if (response_var %in% colnames(new_data)) {
      new_data <- new_data[, !(colnames(new_data) %in% response_var), drop = FALSE]
    }

    # If classification and probability prediction is enabled
    if (!is.null(caret_params$trControl) &&
        isTRUE(caret_params$trControl$classProbs)) {

      # Output of Cram ML: dataframe with probabilities for each class (even if binary)
      predictions <- predict(model, newdata = new_data, type = "prob")

      # if binary, corresponds to proba class = 1
      if (isTRUE(cram_policy_handle)) {
        predictions <- expected_outcome(predictions)
      }

    } else {
      predictions <- predict(model, newdata = new_data)
      # Could be of factor type for Cram ML for classification (output labels)
      # For Cram Policy, for classification, we recommended to use type = prob and not this.
      if (isTRUE(cram_policy_handle)) {
        if (is.factor(predictions)) {
          stop("For classification with cram policy, you must set `classProbs = TRUE` in `trainControl` inside your `caret_params`.")
        }
      }
    }

  } else {
    stop("Error: Model not found or method not supported in caret_params, please set a formula.")
  }

  return(predictions)
}
