#' Cram Estimator for Policy Value (Psi)
#'
#' This function returns the cram estimator for the policy value (psi).
#'
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns,
#'           where n is the sample size and nb_batch is the number of batches,
#'           containing the policy assignment for each individual for each policy.
#'           The first column represents the baseline policy.
#' @param batch_indices A list where each element is a vector of indices corresponding to the individuals in each batch.
#' @return The estimated policy value.
#' @examples
#' # Example usage
#' Y <- sample(0:1, 100, replace = TRUE)
#' D <- sample(0:1, 100, replace = TRUE)
#' pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
#' batch_indices <- split(1:100, rep(1:10, each = 10))
#' estimate <- cram_policy_value_estimator(Y, D, pi, batch_indices)
#' @export

cram_policy_value_estimator <- function(Y, D, pi, batch_indices) {
  # Determine the number of batches
  T <- length(batch_indices)

  # Extract the baseline policy pi0 from the first column of pi
  pi0 <- pi[, 1]

  # Initialize vectors to store psi_j values
  psi_values <- numeric(T)

  # Pre-compute the weights for all individuals
  weight_diff <- Y * (D / 0.5 - (1 - D) / 0.5)

  # Loop through each batch to calculate psi_j
  for (j in 1:T) {
    batch <- batch_indices[[j]]

    # Calculate eta_hat_j for batch j with e(X) = 1/2
    eta_j <- mean(
      Y[batch] * D[batch] * pi0[batch] / 0.5 +
        Y[batch] * (1 - D[batch]) * (1 - pi0[batch]) / 0.5
    )

    # Calculate psi_j based on the value of j
    if (j == 1) {
      # For j = 1, psi_j is simply 1/T * eta_hat_j
      psi_values[j] <- (1 / T) * eta_j
    } else {
      # For j >= 2, calculate Gamma_hat_j(T)
      gamma_j_T <- 0
      for (t in 1:(j - 1)) {
        gamma_tj <- 0
        policy_diff <- pi[batch, t + 1] - pi[batch, t]
        gamma_tj <- mean(weight_diff[batch] * policy_diff)

        # Accumulate gamma_j_T with the weighting factor
        gamma_j_T <- gamma_j_T + (gamma_tj / (T - t))
      }

      # For j >= 2, psi_j is (1/T * eta_hat_j) + Gamma_hat_j(T)
      psi_values[j] <- (1 / T) * eta_j + gamma_j_T
    }
  }

  # Sum all psi_j values to get the overall policy value difference estimate
  policy_value_difference_estimate <- sum(psi_values)

  return(policy_value_difference_estimate)
}
