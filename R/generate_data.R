#' Generate Mock Dataset
#'
#' This function generates a simulated dataset with covariates, treatment assignments,
#' and outcomes for testing and experimentation. The dataset includes heterogeneous
#' treatment effects across groups, mimicking realistic causal inference scenarios.
#'
#' @param n Integer. The number of observations to generate.
#' @return A list containing:
#' \describe{
#'   \item{X}{A \code{data.table} with three variables:}
#'     \describe{
#'       \item{binary}{Binary covariate (0 or 1).}
#'       \item{discrete}{Discrete covariate (values from 1 to 5).}
#'       \item{continuous}{Continuous covariate (normally distributed).}
#'     }
#'   \item{D}{Binary treatment assignment (0 or 1).}
#'   \item{Y}{Numeric outcome based on treatment effects and covariates.}
#' }
#' @examples
#' # Generate a dataset with 1000 observations
#' data <- generate_data(1000)
#' str(data)
#' head(data$X)
#' head(data$D)
#' head(data$Y)
#' @export
generate_data <- function(n) {
  X <- data.table(
    binary = rbinom(n, 1, 0.5),                 # Binary variable (0 or 1)
    discrete = sample(1:5, n, replace = TRUE),  # Discrete variable (1 to 5)
    continuous = rnorm(n)                       # Continuous variable
  )

  # Treatment generation
  D <- rbinom(n, 1, 0.5)  # Binary treatment with 50% probability

  # Outcome generation with heterogeneous treatment effect
  treatment_effect <- ifelse(
    X[["binary"]] == 1 & X[["discrete"]] <= 2,    # Group 1: High benefit
    1,
    ifelse(X[["binary"]] == 0 & X[["discrete"]] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect (small positive or negative)
  )

  # Define the outcome Y
  Y <- D * (treatment_effect + rnorm(n, mean = 0, sd = 1)) +
    (1 - D) * rnorm(n)

  # Return as a list
  return(list(X = X, D = D, Y = Y))
}

