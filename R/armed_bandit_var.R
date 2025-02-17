#' Crammed Variance Estimator
#'
#' This function implements the crammed variance estimator \eqn{\hat{\sigma}^2_T}
#' as described in the provided formula.
#'
#' @param Y A vector of observed rewards for each time step.
#' @param D A vector of treatment indicators (1 if treated, 0 if not) for each time step.
#' @param pi A matrix where each row represents the policy probabilities for actions at time t.
#' @param X A matrix or data frame of covariates observed at each time step.
#' @param T The total number of iterations in the bandit process.
#' @return The crammed variance estimate \eqn{\hat{\sigma}^2_T}.
#' @export

cram_bandit_var <- function(pi, reward, arm) {
  dims_result <- dim(result)

  # Extract relevant dimensions
  nb_arms <- dims_result[3]
  nb_timesteps <- dims_result[2]


  ## POLICY SLICED: remove the arm dimension as Xj is associated to Aj

  # pi:
  # for each row j, column t, depth a, gives pi_t(Xj, a)

  # We do not need the last column and the first row
  # We only need, for each row j, pi_t(Xj, Aj), where Aj is the arm chosen from context j
  # Aj is the jth element of the vector arm, and corresponds to a depth index

  # drop = False to maintain 3D structure

  pi <- pi[-1, -ncol(pi), , drop = FALSE]

  depth_indices <- arm[2:nb_timesteps]

  pi <- extract_2d_from_3d(pi, depth_indices)

  # pi is now a (T-1) x (T-1) matrix

  ## POLICY DIFFERENCE

  pi_diff <- pi[, -1] - pi[, -ncol(pi)]

  # Vector of terms for each column
  policy_diff_weights <- 1 / (nb_timesteps - (1:(nb_timesteps - 1)))

  # pi_diff is a (T-1) x (T-2) matrix





  return(sigma_squared_T)
}
