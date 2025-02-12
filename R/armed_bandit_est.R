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


  ## POLICY SLICED: remove the arm dimension as Xj is associated to Aj

  # pi:
  # for each row j, column t, depth a, gives pi_t(Xj, a)

  # We do not need the last column and the first two rows
  # We only need, for each row j, pi_t(Xj, Aj), where Aj is the arm chosen from context j
  # Aj is the jth element of the vector arm, and corresponds to a depth index

  # drop = False to maintain 3D structure

  pi <- pi[-c(1,2), -ncol(pi), , drop = FALSE]
  depth_indices <- arm[3:nb_timesteps]

  pi <- extract_2d_from_3d(pi, depth_indices)

  # pi is now a (T-2) x (T-1) matrix


  ## POLICY DIFFERENCE

  pi_diff <- pi[, -1] - pi[, -ncol(pi)]

  # pi_diff is a (T-2) x (T-2) matrix


  ## MULTIPLY by Rj / pi_j-1

  # Get diagonal elements from pi (i, i+1 positions)
  pi_diag <- pi[cbind(1:(nrow(pi)), 2:(ncol(pi)))]  # Vectorized indexing

  # Create multipliers using vectorized operations
  multipliers <- (1 / pi_diag) * reward[3:length(reward)]

  # Apply row-wise multiplication using efficient matrix operation
  mult_pi_diff <- pi_diff * multipliers  # Works via R's recycling rules (most efficient)


  ## AVERAGE TRIANGLE INF COLUMN-WISE

  mult_pi_diff[upper.tri(mult_pi_diff, diag = FALSE)] <- NA

  deltas <- colMeans(mult_pi_diff, na.rm = TRUE, dims = 1)  # `dims=1` ensures row-wise efficiency


  ## SUM DELTAS

  sum_deltas <- sum(deltas)


  ## ADD V(pi_1)

  pi_first_col <- pi[, 1]
  pi_first_col <- pi_first_col * multipliers

  # add the term for j = 2, this is only reward 2!
  r2 <- reward[2]
  pi_first_col <- c(pi_first_col, r2)

  # V(pi_1) is the average
  v_pi_1 <-  mean(pi_first_col)


  ## FINAL ESTIMATE

  estimate <- sum_deltas + v_pi_1

  return(estimate)
}
