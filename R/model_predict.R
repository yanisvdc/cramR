#' Predict with the Specified Model
#'
#' This function performs inference using a trained model, providing flexibility for different types of models
#' such as Causal Forest, Ridge Regression, and Feedforward Neural Networks (FNNs).
#'
#' @param model A trained model object returned by the `fit_model` function.
#' @param X A matrix or data frame of covariates for which predictions are required.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for untreated). Optional, depending on the model type.
#' @param model_type A string specifying the model type. Supported options are \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}.
#' @param learner_type A string specifying the learner type. Supported options depend on the \code{model_type}.
#'                     For example, \code{"ridge"} and \code{"fnn"} are supported learners for \code{"s_learner"}.
#' @param model_params A list of additional parameters used during training, required for context-sensitive inference.
#' @return A vector of predictions or CATE estimates, depending on the \code{model_type} and \code{learner_type}.
#' @examples
#' # Example: Predicting with a Causal Forest model
#' cf_model <- causal_forest(X, Y, D)
#' predictions <- model_predict(model = cf_model, X = new_X, model_type = "causal_forest")
#'
#' # Example: Predicting with a Ridge Regression S-learner
#' ridge_model <- cv.glmnet(X_train, Y_train)
#' predictions <- model_predict(model = ridge_model, X = X_test, model_type = "s_learner", learner_type = "ridge")
#'
#' # Example: Predicting with an FNN S-learner
#' fnn_model <- keras_model_sequential()
#' predictions <- model_predict(model = fnn_model, X = X_test, model_type = "s_learner", learner_type = "fnn")
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{predict}}
#' @importFrom grf predict
#' @importFrom glmnet predict
#' @importFrom keras predict
#' @export
model_predict <- function(model, X, D = NULL, model_type, learner_type = NULL, model_params = list()) {
  if (model_type == "causal_forest") {
    # Predict using Causal Forest
    predictions <- grf::predict(model, X)$predictions

  } else if (model_type == "s_learner") {
    if (learner_type == "ridge") {
      # Predict with Ridge Regression
      predictions_treated <- glmnet::predict(model, newx = as.matrix(cbind(X, rep(1, nrow(X)))))
      predictions_control <- glmnet::predict(model, newx = as.matrix(cbind(X, rep(0, nrow(X)))))
      predictions <- predictions_treated - predictions_control

    } else if (learner_type == "fnn") {
      # Predict with Feedforward Neural Network
      treated_input <- as.matrix(cbind(X, rep(1, nrow(X))))
      control_input <- as.matrix(cbind(X, rep(0, nrow(X))))

      predictions_treated <- predict(model, treated_input)
      predictions_control <- predict(model, control_input)

      predictions <- as.numeric(predictions_treated) - as.numeric(predictions_control)

    } else {
      stop("Unsupported learner_type for S-learner. Choose 'ridge' or 'fnn'.")
    }

  } else if (model_type == "m_learner") {
    if (is.null(D)) {
      stop("Error: D (treatment indicators) must be provided for M-learner predictions.")
    }

    # Estimate propensity scores
    propensity_model <- glm(D ~ ., data = as.data.frame(X), family = "binomial")
    prop_scores <- predict(propensity_model, newdata = as.data.frame(X), type = "response")

    if (learner_type == "ridge") {
      # Transformed outcome prediction with Ridge Regression
      Y_star <- as.numeric(model_params$Y_star)
      transformed_outcome <- glmnet::predict(model, newx = as.matrix(X))
      predictions <- transformed_outcome

    } else if (learner_type == "fnn") {
      # Transformed outcome prediction with Feedforward Neural Network
      predictions <- predict(model, as.matrix(X))

    } else {
      stop("Unsupported learner_type for M-learner. Choose 'ridge' or 'fnn'.")
    }

  } else {
    stop("Unsupported model_type. Choose 'causal_forest', 's_learner', or 'm_learner'.")
  }

  return(predictions)
}
