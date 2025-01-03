#' Example Dataset: toydata
#'
#' This dataset is provided as a starting point for users to experiment with the
#' package's functionality. It contains simulated covariates, treatment assignments,
#' and outcomes generated using heterogeneous treatment effects.
#'
#' @docType data
#' @name toydata
#' @usage data(toydata)
#' @export
#'
#' @format A \code{data.table} with 1000 observations and 5 variables:
#' \describe{
#'   \item{binary}{Binary covariate (0 or 1).}
#'   \item{discrete}{Discrete covariate (values from 1 to 5).}
#'   \item{continuous}{Continuous covariate (normally distributed).}
#'   \item{D}{Binary treatment assignment (0 or 1).}
#'   \item{Y}{Numeric outcome based on treatment effects and covariates.}
#' }
#' @keywords datasets
#' @examples
#' data(toydata)
#' str(toydata)
#' head(toydata)
#'
"toydata"
