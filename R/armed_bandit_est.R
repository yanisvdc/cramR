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
