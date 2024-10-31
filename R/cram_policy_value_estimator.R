#' Cram Policy Value Estimator with Gamma Calculation
#'
#' This function estimates the policy value of a learned policy using the cram approach,
#' assuming e(X) = 1/2 and using the first column of pi as pi0.
#' 
#' @param Y A vector of observed outcomes for each individual.
#' @param D A vector of binary treatment assignments (1 for treated, 0 for control).
#' @param pi A matrix where the first column represents the baseline policy (pi0),
#'           and subsequent columns represent policy assignments for each batch.
#' @param batch_indices A list where each element is a vector of indices corresponding to individuals in each batch.
#' @return The estimated policy value difference.
#' @examples
#' # Example usage
#' Y <- sample(0:1, 100, replace = TRUE)
#' D <- sample(0:1, 100, replace = TRUE)
#' pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
#' batch_indices <- split(1:100, rep(1:10, each = 10))
#' estimate <- cram_policy_value_estimator(Y, D, pi, batch_indices)

cram_policy_value_estimator <- function(Y, D, pi, batch_indices) {
  # Determine the number of batches
  T <- length(batch_indices)
  
  # Extract the baseline policy pi0 from the first column of pi
  pi0 <- pi[, 1]
  
  # Initialize vectors to store psi_j values
  psi_values <- numeric(T)
  
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
        for (i in batch) {
          # IPW estimator component for Gamma_hat_{tj} with e(X) = 1/2
          weight_diff <- Y[i] * D[i] / 0.5 - Y[i] * (1 - D[i]) / 0.5
          policy_diff <- pi[i, t + 1] - pi[i, t]
          gamma_tj <- gamma_tj + weight_diff * policy_diff
        }
        # Average over the batch size and apply the weighting factor
        gamma_tj <- gamma_tj / length(batch)
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
