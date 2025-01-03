#' Fit a Provided Model Based on Type and Learner
#'
#' This function trains a given model with the provided data and parameters, taking into account the model type and learner type.
#'
#' @param model An unfitted model object, as returned by `set_model`.
#' @param X A matrix or data frame of covariates for the samples.
#' @param Y A vector of outcome values.
#' @param W (Optional) A vector of binary treatment indicators (for models requiring treatment data, e.g., M-learner, Causal Forest).
#' @param model_type A string specifying the type of model. Supported options: "Causal Forest", "S-learner", "M-learner".
#' @param learner_type A string specifying the type of learner. Supported options: "ridge", "FNN".
#' @param model_params A list of additional parameters to pass to the fitting process.
#' @return The fitted model object.
#' @examples
#' # Example usage for Ridge Regression S-learner
#' set.seed(123)
#' X <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # Covariates (100 samples, 10 features)
#' W <- sample(0:1, 100, replace = TRUE)           # Binary treatment indicators
#' Y <- rnorm(100)                                 # Outcome values
#' # Set up the model
#' model <- set_model("s_learner", "ridge")
#' # Define model parameters
#' model_params <- list(alpha = 0)  # Regularization parameter for Ridge Regression
#' # Fit the model
#' fitted_model <- fit_model(model, X, Y, W = W, model_type = "s_learner", learner_type = "ridge", model_params = model_params)
#' @export
fit_model <- function(model, X, Y, W, model_type, learner_type, model_params) {
  # Validate input
  if (is.null(model)) {
    stop("The provided model is NULL. Please ensure `set_model` returns a valid model.")
  }
  if (!model_type %in% c("causal_forest", "s_learner", "m_learner")) {
    stop("Unsupported model type. Choose 'causal_forest', 's_learner', or 'm_learner'.")
  }
  if (!learner_type %in% c("ridge", "fnn") && model_type != "causal_forest") {
    stop("Unsupported learner type for this model type. Choose 'ridge' or 'fnn'.")
  }
  if (is.null(W)) {
    stop("Treatment indicators (W) are required.")
  }

  fitted_model <- NULL

  if (model_type == "causal_forest") {
    # Train Causal Forest
    fitted_model <- do.call(model, c(list(X = X, Y = Y, W = W), model_params))

  } else if (model_type == "s_learner") {
    if (learner_type == "ridge") {
      # Ridge Regression (S-learner)
      X <- cbind(as.matrix(X), W)  # Add treatment indicator for S-learner

      fitted_model <- do.call(model, c(list(x = as.matrix(X), y = Y), model_params))

    } else if (learner_type == "fnn") {
      # Feedforward Neural Network (S-learner)
      X <- cbind(as.matrix(X), W)  # Add treatment indicator for S-learner

      history <- model %>% fit(
        as.matrix(X),
        Y,
        epochs = model_params$fit_params$epochs,
        batch_size = model_params$fit_params$batch_size,
        verbose = model_params$fit_params$verbose,
        callbacks = NULL
      )
      fitted_model <- model
    }
  } else if (model_type == "m_learner") {
    # M-learner requires a propensity score and transformed outcomes

    # Propensity score estimation
    propensity_model <- glm(W ~ ., data = as.data.frame(X), family = "binomial")
    prop_score <- predict(propensity_model, newdata = as.data.frame(X), type = "response")
    Y_star <- Y * W / prop_score - Y * (1 - W) / (1 - prop_score)

    if (learner_type == "ridge") {
      # Ridge Regression for M-learner
      fitted_model <- do.call(model, c(list(x = as.matrix(X), y = Y_star), model_params))

    } else if (learner_type == "fnn") {
      # Feedforward Neural Network for M-learner
      history <- model %>% fit(
        as.matrix(X),
        Y_star,
        epochs = model_params$fit_params$epochs,
        batch_size = model_params$fit_params$batch_size,
        verbose = model_params$fit_params$verbose
      )
      fitted_model <- model
    }
  }

  return(fitted_model)
}
