#' Simulated Dataset: X_D_Y_example_data
#'
#' This dataset contains covariates, treatment assignments, and outcomes generated
#' using an external script with heterogeneous treatment effects. It is intended
#' for testing and demonstrating causal inference techniques.
#'
#' @docType data
#' @name X_D_Y_example_data
#' @aliases X_D_Y_example_data
#'
#' @usage data(X_D_Y_example_data)
#'
#' @format A list containing:
#' \describe{
#'   \item{X}{A \code{data.table} with three variables:}
#'     \describe{
#'       \item{binary}{Binary covariate (0 or 1).}
#'       \item{discrete}{Discrete covariate (values from 1 to 5).}
#'       \item{continuous}{Continuous covariate (normally distributed).}
#'     }
#'   \item{D}{A binary treatment assignment vector (0 or 1).}
#'   \item{Y}{A numeric outcome vector generated based on treatment effects and covariates.}
#' }
#' @keywords datasets
#' @examples
#' data(X_D_Y_example_data)
#' str(X_D_Y_example_data)
#' head(X_D_Y_example_data$X)
#' head(X_D_Y_example_data$D)
#' head(X_D_Y_example_data$Y)
#'
"X_D_Y_example_data"
