#' Cram Estimator for Policy Value Difference (Delta)
#'
#' This function returns the cram estimator for the policy value difference (delta).
#'
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns,
#'           where n is the sample size and nb_batch is the number of batches,
#'           containing the policy assignment for each individual for each policy.
#'           The first column represents the baseline policy.
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

  # Pre-compute the weights for all individuals
  weight_diff <- Y * (D / 0.5 - (1 - D) / 0.5)

  # Loop through each batch (from j = 2 to T)
  for (j in 2:nb_batch) {
    # Calculate the summand for Gamma_hat_j(T) based on the inner sum over t
    gamma_j_T <- 0
    for (t in 1:(j - 1)) {
      # Compute Gamma_hat_tj for this batch
      gamma_tj <- 0
      policy_diff <- pi[batch_indices[[j]], t + 1] - pi[batch_indices[[j]], t]
      gamma_tj <- mean(weight_diff[batch_indices[[j]]] * policy_diff)

      # Accumulate gamma_j_T with the weighting factor
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
