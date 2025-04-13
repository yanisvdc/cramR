#' Validate User-Provided Parameters for a Model
#'
#' This function validates user-provided parameters against the formal arguments of a specified model function.
#' It ensures that all user-specified parameters are recognized by the model and raises an error for invalid parameters.
#'
#' @param model_function The model function for which parameters are being validated (e.g., \code{grf::causal_forest}).
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param user_params A named list of parameters provided by the user.
#' @return A named list of validated parameters that are safe to pass to the model function.
#' @examples
#' # Example with causal_forest from grf
#' library(grf)
#' set.seed(123)
#' my_X <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # Covariates
#' my_Y <- rnorm(100)                                  # Outcome variable
#' my_W <- sample(0:1, 100, replace = TRUE)            # Binary treatment indicator
#' # Define user parameters
#' user_params <- list(num.trees = 100)
#'
#' # Validate parameters
#' valid_params <- validate_params(grf::causal_forest, "causal_forest", NULL, user_params)
#'
#' # Use the validated parameters to call the model
#' # X, Y, W must still be passed explicitly
#' cf_model <- do.call(grf::causal_forest, c(list(X = my_X, Y = my_Y, W = my_W), valid_params))
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[base]{formals}}
#' @export
validate_params <- function(model_function, model_type, learner_type, user_params) {
  if (is.null(user_params)){
    if (model_type == "causal_forest") {
      default_model_params <- list(num.trees = 100)
      return(default_model_params)
      # return immediately to avoid to overwrite
    } else if (learner_type == "ridge") {
      default_model_params <- list(alpha = 1)
      return(default_model_params)
    } else if (learner_type == "caret") {
      default_model_params <- list(formula = Y ~ ., caret_params = list(method = "lm", trControl = trainControl(method = "none")))
      return(default_model_params)
    }else {
      stop("Error: model_type should be one of 'causal_forest', 's_learner', 'm_learner', and learner_type should be one of 'ridge', 'fnn'.")
    }
  }

  # If the previous test did not return, the user specified model_params
  # Retrieve the full list of arguments for the model function
  formal_args <- formals(model_function)
  valid_args <- names(formal_args)

  # Identify positional arguments (those without default values), excluding ...
  positional_args <- names(formal_args)[
    sapply(formal_args, function(arg) identical(arg, quote(expr = ))) & names(formal_args) != "..."
  ]

  # Check if the function allows additional arguments via ...
  allows_dotdotdot <- "..." %in% valid_args

  if (!allows_dotdotdot) {
    # Exclude positional arguments from validation
    valid_named_args <- setdiff(valid_args, positional_args)

    # Exclude M-learner-specific fields from argument validation
    excluded_keys <- c("m_learner_outcome_transform", "m_learner_propensity_model")
    filtered_user_params <- user_params[setdiff(names(user_params), excluded_keys)]

    # Find invalid user-provided parameters
    invalid_params <- setdiff(names(filtered_user_params), valid_named_args)

    # Raise an error if invalid parameters are found
    if (length(invalid_params) > 0) {
      stop(paste("Invalid parameters for the model:", paste(invalid_params, collapse = ", ")))
    }
  }
  # Return the valid parameters
  return(user_params)
}
