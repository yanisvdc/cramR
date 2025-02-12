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

  # pi:
  # for each row j, column t, depth a, gives pi_t(Xj, a)

  # pi has T columns, we do not need the last one (vs for policy learning, we had nb_batch+1 columns)
  # drop = False to maintain 3D structure
  # we remove the first two rows corresponding to context j=1 and j=2
  # Finally, for row j (corresponding to context Xj)

  pi <- pi[-c(1,2), -ncol(pi), , drop = FALSE]
  pi <- pi[cbind(1:(nb_timesteps-2), 1:(nb_timesteps-1), arm[3:nb_timesteps])]
  # Reshape
  dim(pi) <- c(nb_timesteps-2, nb_timesteps-1)




  # pi_diff <- diff(pi[-c(1,2), 1:(nb_timesteps - 1), , drop = FALSE], differences = 1, lag = 1, dim = 2)


  ## MATRIX OF PI_{j-1} of X_j, for j from 3 to nb_timesteps, across all arms

  # each row of the result matrix corresponds to a j
  # each column of the result matrix corresponds to an arm

  rows <- 3:nb_timesteps    # Indices for row selection
  cols <- rows - 1    # Corresponding column indices
  pi_arm_selection <- pi[cbind(rows, cols), , drop = TRUE]

  # for each row i of the matrix, we retain the column of index arm[i+2]
  # for i from 1 to T-2, i=1 corresponds to j=3 (see how we got the matrix)
  # so the row corresponds to pi_2(X_3, a), for a in 1, .., K
  # arm[i+2] is arm[3] in this case, and corresponds to what pi_2 ended up choosing
  # which is not necessarily the arm with highest probability
  # it is the arm that maximized the linear part of the reward using the
  # arm-specific parameters sampled by pi_2 for each arm and using the context X_3

  # This is a vector containing the probability that the arm chosen
  # had to be chosen given the current policy and the context at time t
  pi_arm_chosen <- pi_arm_selection[cbind(1:(nb_timesteps-2), arm[3:nb_timesteps]), drop = TRUE]


  ## PI DIFF FOR CHOSEN ARM

  # Use advanced indexing to extract selected depth values
  pi_diff <- A[cbind(rows, cols, depth_indices)]

























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










extract_2d_from_3d <- function(array3d, depth_indices) {
  # Get array dimensions
  dims <- dim(array3d)
  nrow <- dims[1]  # Rows
  ncol <- dims[2]  # Columns

  # Ensure depth_indices length matches required rows
  if (length(depth_indices) != nrow) {
    stop("Depth indices vector must have exactly (T-2) elements.")
  }

  # Vectorized index calculation
  i <- rep(1:nrow, each = ncol)  # Row indices
  j <- rep(1:ncol, times = nrow) # Column indices
  k <- rep(depth_indices, each = ncol)  # Depth indices

  # Calculate linear indices for efficient extraction
  linear_indices <- i + (j - 1) * nrow + (k - 1) * nrow * ncol

  # Create result matrix using vectorized indexing
  result_matrix <- matrix(array3d[linear_indices], nrow = nrow, ncol = ncol, byrow = TRUE)

  return(result_matrix)
}

# # Define parameters
# T <- 5   # Total time points
# K <- 2   # Number of depth layers
#
# # Create a 3D array (T-2 x T-1 x K)
# array3d <- array(1:((T-2)*(T-1)*K), dim = c(T-2, T-1, K))
#
# # Create arm vector (must contain indices between 1 and K)
# arm <- c(1, 2, 1, 2, 1)  # Example arm vector
#
# # Vectorized index calculation
# depth_indices <- arm[3:T]  # Select arm[i+2] values
#
# res <- extract_2d_from_3d(array3d, depth_indices)







# # Define new test parameters
# T_new <- 6   # New total time points
# K_new <- 3   # New number of depth layers
#
# # Create a new 3D array (T-2 x T-1 x K)
# array3d_new <- array(1:((T_new-2)*(T_new-1)*K_new), dim = c(T_new-2, T_new-1, K_new))
#
# # Create a new arm vector (must contain indices between 1 and K)
# arm_new <- c(2, 3, 1, 2, 3, 1)  # Example arm vector
#
# # Extract depth indices from arm
# depth_indices_new <- arm_new[3:T_new]  # Select arm[i+2] values
#
# # Apply function to new data
# result_new <- extract_2d_from_3d(array3d_new, depth_indices_new)
#
# # Output results
# cat("\nNew 3D Array:\n")
# print(array3d_new)
#
# cat("\nNew Arm Vector:\n")
# print(arm_new)
#
# cat("\nNew Depth Indices:\n")
# print(depth_indices_new)
#
# cat("\nExtracted 2D Matrix:\n")
# print(result_new)
#
