#' Validate User-Provided Parameters for a Model
#'
#' This function validates user-provided parameters against the formal arguments of a specified model function.
#' It ensures that all user-specified parameters are recognized by the model and raises an error for invalid parameters.
#'
#' @param model_function The model function for which parameters are being validated (e.g., \code{grf::causal_forest}).
#' @param user_params A named list of parameters provided by the user.
#' @param positional_args A character vector of arguments that must be provided explicitly and should not be validated (e.g., \code{c("X", "Y", "W")}). Default is \code{c("X", "Y", "W")}.
#' @return A named list of validated parameters that are safe to pass to the model function.
#' @examples
#' # Example with causal_forest from grf
#' library(grf)
#'
#' # Define user parameters
#' user_params <- list(num.trees = 1000, sample.fraction = 0.8)
#'
#' # Validate parameters
#' valid_params <- validate_params(grf::causal_forest, user_params)
#'
#' # Use the validated parameters to call the model
#' # X, Y, W must still be passed explicitly
#' causal_forest(X = my_X, Y = my_Y, W = my_W, !!!valid_params)
#'
#' # Invalid parameters example
#' user_params <- list(nonexistent_param = 42)  # Invalid parameter
#' validate_params(grf::causal_forest, user_params)
#' # Error: Invalid parameters for the model: nonexistent_param
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[base]{formals}}
#' @importFrom base formals setdiff
#' @export
validate_params <- function(model_function, user_params, positional_args = c("X", "Y", "W")) {
  # Retrieve the full list of arguments for the model function
  valid_args <- names(formals(model_function))

  # Exclude positional arguments from validation
  valid_named_args <- setdiff(valid_args, positional_args)

  # Find invalid user-provided parameters
  invalid_params <- setdiff(names(user_params), valid_named_args)

  # Raise an error if invalid parameters are found
  if (length(invalid_params) > 0) {
    stop(paste("Invalid parameters for the model:", paste(invalid_params, collapse = ", ")))
  }

  # Return the valid parameters
  user_params
}
