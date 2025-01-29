#' Cramming Policy Evaluation for Multi-Armed Bandit
#'
#' This function implements the armed bandit policy evaluation formula for
#' estimating \eqn{\Delta(\pi_T; \pi_0)} as given in the user-provided formula.
#'
#' @param policy_diff A matrix where each entry represents the difference in policies
#'        between iterations, i.e., \eqn{\pi_t(X, a) - \pi_{t-1}(X, a)} for each t.
#' @param reward A matrix of observed rewards corresponding to each action and time step.
#' @param T The total number of iterations in the bandit process.
#' @return The estimated policy value difference \eqn{\Delta(\pi_T; \pi_0)}.
#' @examples
#' # Example usage
#' set.seed(123)
#' T <- 10
#' K <- 5  # Number of actions
#'
#' # Simulate policy differences and rewards
#' policy_diff <- matrix(runif(T * K, -0.5, 0.5), nrow = T, ncol = K)
#' reward <- matrix(rnorm(T * K), nrow = T, ncol = K)
#'
#' cram_bandit_policy_eval(policy_diff, reward, T)
#' @export

cram_bandit_policy_eval <- function(policy_diff, reward, T) {
  if (!is.matrix(policy_diff) || !is.matrix(reward)) {
    stop("Both `policy_diff` and `reward` must be matrices.")
  }
  if (nrow(policy_diff) != T || nrow(reward) != T) {
    stop("Both `policy_diff` and `reward` must have `T` rows.")
  }

  # Initialize the overall policy difference estimate
  delta_estimate <- 0

  # Compute the sum of weighted policy differences
  for (j in 2:T) {
    gamma_j_T <- 0
    for (t in 1:(j - 1)) {
      weight <- 1 / (T - t)
      gamma_j_T <- gamma_j_T + weight * sum(policy_diff[t, ] * reward[j, ])
    }
    delta_estimate <- delta_estimate + gamma_j_T
  }

  return(delta_estimate)
}
