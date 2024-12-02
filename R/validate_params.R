#' Validate User-Provided Parameters for a Model
#'
#' This function validates user-provided parameters against the formal arguments of a specified model function.
#' It ensures that all user-specified parameters are recognized by the model and raises an error for invalid parameters.
#'
#' @param model_function The model function for which parameters are being validated (e.g., \code{grf::causal_forest}).
#' @param user_params A named list of parameters provided by the user.
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
validate_params <- function(model_function, user_params) {
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

    # Find invalid user-provided parameters
    invalid_params <- setdiff(names(user_params), valid_named_args)

    # Raise an error if invalid parameters are found
    if (length(invalid_params) > 0) {
      stop(paste("Invalid parameters for the model:", paste(invalid_params, collapse = ", ")))
    }

  } else {
      # If ... is present, assume all user-provided parameters are allowed
      message("The function accepts additional parameters via '...'. Assume that all user-provided parameters are allowed.")
  }

  # Return the valid parameters
  return(user_params)
}
