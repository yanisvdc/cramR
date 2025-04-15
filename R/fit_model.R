#' Fit Model
#'
#' This function trains a given unfitted model with the provided data and parameters,
#' according to model type and learner type.
#'
#' @param model An unfitted model object, as returned by `set_model`.
#' @param X A matrix or data frame of covariates for the samples.
#' @param Y A vector of outcome values.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for untreated).
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @return The fitted model object.
#' @examples
#' # Example usage for Ridge Regression S-learner
#' set.seed(123)
#' X <- matrix(rnorm(1000), nrow = 100, ncol = 10)
#' D <- sample(0:1, 100, replace = TRUE)
#' Y <- rnorm(100)
#' # Set up the model
#' model <- set_model("s_learner", "ridge")
#' # Define model parameters
#' model_params <- list(alpha = 0)
#' # Fit the model
#' fitted_model <- fit_model(
#'                         model, X, Y, D = D,
#'                         model_type = "s_learner",
#'                         learner_type = "ridge",
#'                         model_params = model_params)
#' @export
fit_model <- function(model, X, Y, D, model_type, learner_type, model_params) {
  # Validate input
  if (is.null(model)) {
    stop("The provided model is NULL. Please ensure `set_model` returns a valid model.")
  }
  if (!model_type %in% c("causal_forest", "s_learner", "m_learner")) {
    stop("Unsupported model type. Choose 'causal_forest', 's_learner', or 'm_learner'.")
  }
  if (!learner_type %in% c("ridge", "fnn", "caret") && model_type != "causal_forest") {
    stop("Unsupported learner type for this model type. Choose 'ridge', 'fnn' or 'caret'.")
  }
  if (is.null(D)) {
    stop("Treatment indicators (D) are required.")
  }

  fitted_model <- NULL

  # CAUSAL FOREST -------------------------------------------------------------------
  if (model_type == "causal_forest") {
    # Train Causal Forest
    fitted_model <- do.call(model, c(list(X = X, Y = Y, W = D), model_params))

  # S-LEARNER -----------------------------------------------------------------------
  } else if (model_type == "s_learner") {

    # RIDGE -----------------------------------------------------------------
    if (learner_type == "ridge") {
      # Ridge Regression (S-learner)
      X <- cbind(as.matrix(X), D)  # Add treatment indicator for S-learner

      fitted_model <- do.call(model, c(list(x = as.matrix(X), y = Y), model_params))

    # FNN --------------------------------------------------------------------
    } else if (learner_type == "fnn") {
      # Feedforward Neural Network (S-learner)
      X <- cbind(as.matrix(X), D)  # Add treatment indicator for S-learner

      history <- model %>% fit(
        as.matrix(X),
        Y,
        epochs = model_params$fit_params$epochs,
        batch_size = model_params$fit_params$batch_size,
        verbose = model_params$fit_params$verbose,
        callbacks = NULL
      )
      fitted_model <- model

    # CARET -----------------------------------------------------------------
    } else if (learner_type == "caret") {
      # Caret (S-learner)
      # Ensure X is a data.frame for formula-based caret training
      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
      }
      formula <- model_params$formula
      caret_params <- model_params$caret_params

      # Ensure `formula` is provided
      if (is.null(formula)) {
        stop("Error: A formula must be provided for model training.")
      }

      # Ensure `method` is specified
      if (!"method" %in% names(caret_params)) {
        stop("Error: 'method' must be specified in caret_params.")
      }

      # Set default trControl if not provided
      if (!"trControl" %in% names(caret_params)) {
        caret_params$trControl <- caret::trainControl(method = "none")  # Default to no resampling
      }

      # CARET CLASSIFICATION - Convert 0 & 1 into factors
      if (is_classification_method(caret_params$method)) {
        unique_vals <- unique(Y)

        if (length(unique_vals) == 2 && all(unique_vals %in% c(0, 1))) {
          Y <- factor(Y, levels = c(0, 1), labels = c("no", "yes"))
        }
      }

      X$D <- D # Add treatment indicator for S-learner
      X$Y <- Y # caret uses a formula so we need to add Y to the data

      # Call caret::train() with correctly formatted parameters
      fitted_model <- do.call(model, c(list(formula, data = X), caret_params))
    }

  # M-LEANRER ------------------------------------------------------------------------------
  } else if (model_type == "m_learner") {
    # M-learner requires a propensity model and transformed outcomes
    outcome_transform <- model_params$m_learner_outcome_transform
    propensity_model <- model_params$m_learner_propensity_model

    # PROP SCORE - If no function provided, use default logistic regression-based scorer
    if (is.null(propensity_model)) {
      propensity_model <- function(D, X) {
        p_model <- glm(D ~ ., data = as.data.frame(X), family = "binomial")
        predict(p_model, newdata = as.data.frame(X), type = "response")
      }
    }
    # User-supplied or default prop score
    prop_score <- propensity_model(D, X)

    # OUTCOME TRANSOFRMATION - If not provided, perform IPW difference
    if (is.null(outcome_transform)) {
      outcome_transform <- function(Y, D, prop_score) {
        Y * D / prop_score - Y * (1 - D) / (1 - prop_score)
      }
    }
    # User-supplied or default transformed outcome
    Y <- outcome_transform(Y, D, prop_score)

    # RIDGE --------------------------------------------------------
    if (learner_type == "ridge") {
      # Ridge Regression for M-learner
      fitted_model <- do.call(model, c(list(x = as.matrix(X), y = Y), model_params))

    # FNN ----------------------------------------------------------
    } else if (learner_type == "fnn") {
      # Feedforward Neural Network for M-learner
      history <- model %>% fit(
        as.matrix(X),
        Y,
        epochs = model_params$fit_params$epochs,
        batch_size = model_params$fit_params$batch_size,
        verbose = model_params$fit_params$verbose
      )
      fitted_model <- model

    # CARET --------------------------------------------------------
    } else if (learner_type == "caret") {
      # Caret (M-learner)
      # Ensure X is a data.frame for formula-based caret training
      if (!is.data.frame(X)) {
        X <- as.data.frame(X)
      }
      formula <- model_params$formula
      caret_params <- model_params$caret_params

      # Ensure `formula` is provided
      if (is.null(formula)) {
        stop("Error: A formula must be provided for model training.")
      }

      # Ensure `method` is specified
      if (!"method" %in% names(caret_params)) {
        stop("Error: 'method' must be specified in caret_params.")
      }

      # Set default trControl if not provided
      if (!"trControl" %in% names(caret_params)) {
        caret_params$trControl <- caret::trainControl(method = "none")  # Default to no resampling
      }

      # CARET CLASSIFICATION - Convert 0 & 1 into factors
      if (is_classification_method(caret_params$method)) {
        unique_vals <- unique(Y)

        if (length(unique_vals) == 2 && all(unique_vals %in% c(0, 1))) {
          Y <- factor(Y, levels = c(0, 1), labels = c("no", "yes"))
        }
      }

      X$Y <- Y # caret uses a formula so we need to add Y to the data

      # Call caret::train() with correctly formatted parameters
      fitted_model <- do.call(model, c(list(formula, data = X), caret_params))
    }
  }

  return(fitted_model)
}
