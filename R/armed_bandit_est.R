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
#' @export

cram_bandit_est <- function(pi, reward, arm) {

  dims_result <- dim(result)

  # Extract relevant dimensions
  nb_arms <- dims_result[3]
  nb_timesteps <- dims_result[2]

  ## POLICY DIFFERENCE along the column axis

  # pi has T columns, we do not need the last one (vs for policy learning, we had nb_batch+1 columns)
  # drop = False to maintain 3D structure

  pi_diff <- diff(pi[, 1:(nb_timesteps - 1), , drop = FALSE], differences = 1, lag = 1, dim = 2)


  ## MATRIX OF PI_{j-1} of X_j, for j from 3 to nb_timesteps

  # each row of the result matrix corresponds to a j
  # each column of the result matrix corresponds to an arm

  rows <- 3:nb_timesteps    # Indices for row selection
  cols <- rows - 1    # Corresponding column indices
  pi_arm_selection <- pi[cbind(rows, cols), , drop = FALSE]

















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







# T <- 1000  # Large T
# K <- 500   # Large K
#
# # User-friendly input: A 3D array (T x T x K)
# set.seed(123)
# A <- array(runif(T * T * K), dim = c(T, T, K))
#
# # Compute consecutive column differences directly in the array
# result <- diff(A, differences = 1, lag = 1, dim = 2)  # Efficient and memory-friendly
