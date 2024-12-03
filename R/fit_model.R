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
#' model <- set_model("S-learner", "ridge")
#' fitted_model <- fit_model(model, X, Y, W = D, "S-learner", "ridge", model_params = list(alpha = 0))
#'
#' # Example usage for FNN S-learner
#' model <- set_model("S-learner", "FNN")
#' fitted_model <- fit_model(model, X, Y, W = D, "S-learner", "FNN", model_params = list(epochs = 20))
#'
#' @export
fit_model <- function(model, X, Y, W = NULL, model_type, learner_type, model_params = list()) {
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

  fitted_model <- NULL

  if (model_type == "causal_forest") {
    # Causal Forest requires W (treatment indicator)
    if (is.null(W)) {
      stop("Treatment indicators (W) are required for Causal Forest.")
    }
    # Train Causal Forest
    # fitted_model <- model(X, Y, W, !!!model_params)
    fitted_model <- do.call(model, c(list(X = X, Y = Y, W = W), model_params))

  } else if (model_type == "s_learner") {
    if (learner_type == "ridge") {
      # Ridge Regression (S-learner)
      if (!is.null(W)) {
        X <- cbind(X, W)  # Add treatment indicator for S-learner
      }
      # fitted_model <- model(as.matrix(X), Y, !!!model_params)
      fitted_model <- do.call(model, c(list(X = as.matrix(X), Y = Y), model_params))

    } else if (learner_type == "fnn") {
      # Feedforward Neural Network (S-learner)
      if (!is.null(W)) {
        X <- cbind(X, W)  # Add treatment indicator for S-learner
      }

      fitted_model <- model %>% fit(
        as.matrix(X),
        Y,
        epochs = model_params$epochs,
        batch_size = model_params$batch_size,
        verbose = model_params$verbose
      )
    }
  } else if (model_type == "m_learner") {
    # M-learner requires a propensity score and transformed outcomes
    if (is.null(W)) {
      stop("Treatment indicators (W) are required for M-learner.")
    }

    # Propensity score estimation
    propensity_model <- glm(W ~ ., data = as.data.frame(X), family = "binomial")
    prop_score <- predict(propensity_model, newdata = as.data.frame(X), type = "response")
    Y_star <- Y * W / prop_score - Y * (1 - W) / (1 - prop_score)

    if (learner_type == "ridge") {
      # Ridge Regression for M-learner
      # fitted_model <- model(as.matrix(X), Y_star, !!!model_params)
      fitted_model <- do.call(model, c(list(X = as.matrix(X), Y = Y_star), model_params))

    } else if (learner_type == "fnn") {
      # Feedforward Neural Network for M-learner
      # Set default parameters if not provided
      if (!"epochs" %in% names(model_params)) {
        model_params$epochs <- 10
      }
      if (!"batch_size" %in% names(model_params)) {
        model_params$batch_size <- 32
      }
      if (!"verbose" %in% names(model_params)) {
        model_params$verbose <- 0
      }

      fitted_model <- model %>% fit(
        as.matrix(X),
        Y_star,
        epochs = model_params$epochs,
        batch_size = model_params$batch_size,
        verbose = model_params$verbose
      )
    }
  }

  return(fitted_model)
}
