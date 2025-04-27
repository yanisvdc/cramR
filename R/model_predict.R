#' Cram Policy: Predict with the Specified Model
#'
#' This function performs inference using a trained model, providing flexibility for different types of models
#' such as Causal Forest, Ridge Regression, and Feedforward Neural Networks (FNNs).
#'
#' @param model A trained model object returned by the `fit_model` function.
#' @param X A matrix or data frame of covariates for which predictions are required.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for untreated). Optional, depending on the model type.
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @return A vector of binary policy assignments, depending on the \code{model_type} and \code{learner_type}.
#' @importFrom grf causal_forest
#' @import glmnet
#' @import keras
#' @export
model_predict <- function(model, X, D, model_type, learner_type, model_params) {
  if (model_type == "causal_forest") {
    # Predict using Causal Forest
    predictions <- predict(model, X)$predictions

  } else if (model_type == "s_learner") {
    if (learner_type == "ridge") {
      # Predict with Ridge Regression
      predictions_treated <- predict(model, newx = as.matrix(cbind(X, rep(1, nrow(X)))), s = "lambda.min")
      predictions_control <- predict(model, newx = as.matrix(cbind(X, rep(0, nrow(X)))), s = "lambda.min")
      predictions <- predictions_treated - predictions_control

    } else if (learner_type == "fnn") {
      # Predict with Feedforward Neural Network
      treated_input <- as.matrix(cbind(X, rep(1, nrow(X))))
      control_input <- as.matrix(cbind(X, rep(0, nrow(X))))

      predictions_treated <- predict(model, treated_input)
      predictions_control <- predict(model, control_input)

      predictions <- as.numeric(predictions_treated) - as.numeric(predictions_control)

    } else if (learner_type == "caret") {
      # Predict with Caret
      formula <- model_params$formula
      caret_params <- model_params$caret_params

      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
      }

      # Create treated and control input data.frames properly
      treated_input <- X
      treated_input$D <- 1  # simulate treatment

      control_input <- X
      control_input$D <- 0  # simulate control

      predictions_treated <- model_predict_ml(model, treated_input, formula, caret_params, cram_policy_handle=TRUE)
      predictions_control <- model_predict_ml(model, control_input, formula, caret_params, cram_policy_handle=TRUE)

      predictions <- as.numeric(predictions_treated) - as.numeric(predictions_control)

    } else {
      stop("Unsupported learner_type for S-learner. Choose 'ridge', 'fnn' or 'caret'.")
    }

  } else if (model_type == "m_learner") {

    if (learner_type == "ridge") {
      # Transformed outcome prediction with Ridge Regression
      predictions <- predict(model, newx = as.matrix(X), s = "lambda.min")

    } else if (learner_type == "fnn") {
      # Transformed outcome prediction with Feedforward Neural Network
      predictions <- predict(model, as.matrix(X))

    } else if (learner_type == "caret") {
      # Transformed outcome prediction with Caret
      formula <- model_params$formula
      caret_params <- model_params$caret_params

      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
      }

      predictions <- model_predict_ml(model, X, formula, caret_params, cram_policy_handle=TRUE)

    } else {
      stop("Unsupported learner_type for M-learner. Choose 'ridge', 'fnn' or 'caret'.")
    }

  } else {
    stop("Unsupported model_type. Choose 'causal_forest', 's_learner', or 'm_learner'.")
  }

  # Transform CATEs into a binary policy
  predictions <- as.numeric(predictions)
  predictions <- ifelse(predictions > 0, 1, 0)

  return(predictions)
}
