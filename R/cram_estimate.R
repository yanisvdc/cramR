#' Cram Estimator for Policy Evaluation
#'
#' This function estimates the difference in policy performance between a series of learned policies
#' and a baseline policy using the cram approach.
#'
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns, containing the policy assignment for each individual
#'           for each policy. The first column represents the baseline policy.
#' @param batch_indices A list where each element is a vector of indices corresponding to the individuals in each batch.
#' @return The estimated policy value difference \(\eqn{\hat{\Delta}}(\eqn{\hat{\pi}}_T; \eqn{\pi}_0)\).
#' @examples
#' # Example usage:
#' Y <- sample(0:1, 100, replace = TRUE)
#' D <- sample(0:1, 100, replace = TRUE)
#' pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
#' nb_batch <- 10
#' batch_indices <- split(1:100, rep(1:nb_batch, each = 10))
#' estimate <- cram_estimator(Y, D, pi, batch_indices)
#' @export
cram_estimator <- function(Y, D, pi, batch_indices) {
  # Determine number of batches
  nb_batch <- length(batch_indices)

  # Ensure the number of rows in pi matches the length of Y and D
  if (nrow(pi) != length(Y) || length(Y) != length(D)) {
    stop("Y, D, and pi must have matching lengths")
  }

  # Initialize the policy value difference estimator
  delta_hat <- 0

  # Loop through each batch (from j = 2 to T)
  for (j in 2:nb_batch) {
    # Calculate the summand for Gamma_hat_j(T) based on the inner sum over t
    gamma_j_T <- 0
    for (t in 1:(j - 1)) {
      # Compute Gamma_hat_tj for this batch
      gamma_tj <- 0
      for (i in batch_indices[[j]]) {
        # IPW estimator component
        weight_diff <- Y[i] * D[i] / 0.5 - Y[i] * (1 - D[i]) / 0.5
        policy_diff <- pi[i, t + 1] - pi[i, t]
        gamma_tj <- gamma_tj + weight_diff * policy_diff
      }
      # Average over batch size
      gamma_tj <- gamma_tj / length(batch_indices[[j]])
      # Weighting factor (1 / (T - t))
      gamma_j_T <- gamma_j_T + (gamma_tj / (nb_batch - t))
    }
    # Add Gamma_hat_j(T) to the final estimator
    delta_hat <- delta_hat + gamma_j_T
  }

  return(delta_hat)
}


# Y <- sample(0:1, 100, replace = TRUE)
# D <- sample(0:1, 100, replace = TRUE)
# pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
# batch_indices <- split(1:100, rep(1:nb_batch, each = 10))
# estimate <- cram_estimator(Y, D, pi, batch_indices)
# print(estimate)
