#' Generate Sample Data with Heterogeneous Treatment Effects
#'
#' This function generates synthetic data with heterogeneous treatment effects across different groups.
#' It creates a dataset with covariates, binary treatment assignments, and outcomes based on group-specific treatment effects.
#'
#' @param n Integer specifying the number of samples to generate.
#' @return A list containing:
#'   \describe{
#'     \item{X}{A matrix of covariates with three columns: a binary variable, a discrete variable with values from 1 to 5, and a continuous variable.}
#'     \item{D}{A binary treatment vector of length \code{n}, where each element represents treatment (1) or control (0).}
#'     \item{Y}{A vector of outcomes of length \code{n}, influenced by treatment effects and random noise.}
#'   }
#' @details
#' The covariates in \code{X} include:
#' \itemize{
#'   \item A binary variable with values 0 or 1.
#'   \item A discrete variable with integer values from 1 to 5.
#'   \item A continuous variable from a normal distribution.
#' }
#'
#' The treatment effect is heterogeneous and is assigned as follows:
#' \itemize{
#'   \item High benefit for samples with \code{binary = 1} and \code{discrete <= 2}.
#'   \item Neutral effect (close to zero) for other combinations.
#'   \item High adverse effect for samples with \code{binary = 0} and \code{discrete >= 4}.
#' }
#'
#' The outcome \code{Y} is influenced by the treatment effect if treated (D = 1) and random noise.
#'
#' @examples
#' # Generate data with 100 samples
#' data <- generate_data(100)
#' str(data)  # Check the structure of the generated data
#' @export

# Function to generate sample data with heterogeneous treatment effects: positive, neutral, and adverse
generate_data <- function(n) {
  # Create X with one binary, one discrete, and one continuous variable
  X <- data.frame(
    binary = rbinom(n, 1, 0.5),                # Binary variable (0 or 1)
    discrete = sample(1:5, n, replace = TRUE),  # Discrete variable (integer values from 1 to 5)
    continuous = rnorm(n)                       # Continuous variable (normal distribution)
  )

  # Convert X to matrix form for grf package compatibility
  X <- as.matrix(X)

  # Treatment generation
  D <- rbinom(n, 1, 0.5)  # Binary treatment with 50% probability

  # Outcome generation with heterogeneous treatment effect
  # Define treatment effect for each group
  # - Group 1: High benefit (positive effect)
  # - Group 2: Neutral effect (small effect, around zero)
  # - Group 3: High adverse effect (negative effect)

  treatment_effect <- ifelse(
    X[, "binary"] == 1 & X[, "discrete"] <= 2,    # Group 1: High benefit
    1,
    ifelse(X[, "binary"] == 0 & X[, "discrete"] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect (small positive or negative)
  )

  # Define the outcome Y
  # Outcome depends on the treatment effect and random noise
  Y <- D * (treatment_effect + rnorm(n, mean = 0, sd = 1)) +
    (1 - D) * rnorm(n)  # Outcome influenced by treatment and noise for untreated

  # Return as a list
  return(list(X = X, D = D, Y = Y))
}
