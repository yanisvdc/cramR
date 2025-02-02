#' Fit Model ML
#'
#' This function trains a given unfitted model with the provided data and parameters,
#' according to model type and learner type.
#'
#' @param data The dataset
#' @param formula The formula
#' @param caret_params The parameters for caret model
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
fit_model_ml <- function(data, formula, caret_params) {

  caret_params$formula <- formula
  caret_params$data <- data

  # Train the model using do.call
  fitted_model <- do.call(train, caret_params)

  return(fitted_model)
}
