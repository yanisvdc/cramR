#' Cram Estimator for Policy Value Difference (Delta)
#'
#' This function returns the cram estimator for the policy value difference (delta).
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns,
#'           where n is the sample size and nb_batch is the number of batches,
#'           containing the policy assignment for each individual for each policy.
#'           The first column represents the baseline policy.
#' @param batch_indices A list where each element is a vector of indices corresponding to the individuals in each batch.
#' @param propensity The propensity score function
#' @return The estimated policy value difference \(\eqn{\hat{\Delta}}(\eqn{\hat{\pi}}_T; \eqn{\pi}_0)\).
#' @export
cram_estimator <- function(X, Y, D, pi, batch_indices, propensity = NULL) {
  # Determine number of batches
  nb_batch <- length(batch_indices)

  # Ensure the number of rows in pi matches the length of Y and D
  if (nrow(pi) != length(Y) || length(Y) != length(D)) {
    stop("Y, D, and pi must have matching lengths")
  }

  # Ensure D is binary
  if (!all(D %in% c(0, 1))) {
    stop("D must be a binary vector containing only 0 and 1")
  }

  if (is.null(propensity)) {
    # IPW component for all individuals using default propensity = 0.5
    weight_diff <- Y * D / 0.5 - Y * (1 - D) / 0.5
  } else {
    # Compute propensity scores
    propensity_scores <- propensity(X)

    # Compute IPW component with custom propensity scores
    weight_diff <- Y * D / propensity_scores - Y * (1 - D) / propensity_scores
  }

  policy_diff <- pi[, 2:nb_batch] - pi[, 1:(nb_batch - 1)]

  # Vector of terms for each column
  policy_diff_weights <- 1 / (nb_batch - (1:(nb_batch - 1)))

  # Multiply each column of policy_diff by corresponding policy_diff_weight
  policy_diff <- sweep(policy_diff, 2, policy_diff_weights, FUN = "*")

  policy_diff <- t(apply(policy_diff, 1, cumsum))

  # Create the mask for batch indices
  mask <- matrix(NA, nrow = nrow(policy_diff), ncol = ncol(policy_diff))

  for (k in 2:nb_batch) {
    # Set to NA the rows corresponding to batches 1 to k in column k
    mask[unlist(batch_indices[[k]]), k-1] <- 1
  }

  # Apply the mask to policy_diff
  policy_diff <- policy_diff * mask

  policy_diff <- sweep(policy_diff, 1, weight_diff, FUN = "*")

  # Calculate average for each column
  column_averages <- apply(policy_diff, 2, function(x) mean(x, na.rm = TRUE))

  delta_hat <- sum(column_averages)

  # # Initialize the policy value difference estimator
  # delta_hat <- 0
  #
  # # Pre-compute the weights for all individuals
  # weight_diff <- Y * (D / 0.5 - (1 - D) / 0.5)
  #
  # # Loop through each batch (from j = 2 to T)
  # for (j in 2:nb_batch) {
  #   # Calculate the summand for Gamma_hat_j(T) based on the inner sum over t
  #   gamma_j_T <- 0
  #   for (t in 1:(j - 1)) {
  #     # Compute Gamma_hat_tj for this batch
  #     gamma_tj <- 0
  #     policy_diff <- pi[batch_indices[[j]], t + 1] - pi[batch_indices[[j]], t]
  #     gamma_tj <- mean(weight_diff[batch_indices[[j]]] * policy_diff)
  #
  #     # Accumulate gamma_j_T with the weighting factor
  #     gamma_j_T <- gamma_j_T + (gamma_tj / (nb_batch - t))
  #   }
  #   # Add Gamma_hat_j(T) to the final estimator
  #   delta_hat <- delta_hat + gamma_j_T
  # }

  return(delta_hat)
}










# validate_cram_estimator <- function(Y, D, pi, batch_indices) {
#   # Number of batches
#   nb_batch <- length(batch_indices)
#
#   ### Current Code ###
#   # IPW component for all individuals
#   weight_diff <- Y * (D / 0.5 - (1 - D) / 0.5)
#
#   policy_diff <- pi[, 2:nb_batch] - pi[, 1:(nb_batch - 1)]
#
#   # Vector of terms for each column
#   policy_diff_weights <- 1 / (nb_batch - (1:(nb_batch - 1)))
#   print("Policy Diff Weights:")
#   print(policy_diff_weights)
#
#   # Multiply each column of policy_diff by corresponding policy_diff_weight
#   policy_diff <- sweep(policy_diff, 2, policy_diff_weights, FUN = "*")
#   print("Policy Diff after applying weights:")
#   print(policy_diff)
#
#   policy_diff <- t(apply(policy_diff, 1, cumsum))
#   print("Policy Diff after cumulative sum:")
#   print(policy_diff)
#
#   policy_diff <- sweep(policy_diff, 1, weight_diff, FUN = "*")
#   print("Policy Diff after scaling by weight_diff:")
#   print(policy_diff)
#
#   # Create the mask for batch indices
#   mask <- matrix(NA, nrow = nrow(policy_diff), ncol = ncol(policy_diff))
#
#   for (k in 2:nb_batch) {
#     mask[unlist(batch_indices[[k]]), k-1] <- 1
#   }
#   print("Mask:")
#   print(mask)
#
#   # Apply the mask to policy_diff
#   policy_diff <- policy_diff * mask
#   print("Policy Diff after applying mask:")
#   print(policy_diff)
#
#   # Calculate average for each column
#   column_averages <- apply(policy_diff, 2, function(x) mean(x, na.rm = TRUE))
#   print("Column Averages (Current Code):")
#   print(column_averages)
#
#   delta_hat_current <- sum(column_averages)
#
#   ### Commented-Out Code ###
#   delta_hat_commented <- 0
#
#   # Pre-compute the weights for all individuals
#   weight_diff <- Y * (D / 0.5 - (1 - D) / 0.5)
#
#   # Loop through each batch (from j = 2 to T)
#   for (j in 2:nb_batch) {
#     # Calculate the summand for Gamma_hat_j(T) based on the inner sum over t
#     gamma_j_T <- 0
#     for (t in 1:(j - 1)) {
#       # Compute Gamma_hat_tj for this batch
#       gamma_tj <- 0
#       policy_diff <- pi[batch_indices[[j]], t + 1] - pi[batch_indices[[j]], t]
#       gamma_tj <- mean(weight_diff[batch_indices[[j]]] * policy_diff)
#       gamma_j_T <- gamma_j_T + (gamma_tj / (nb_batch - t))
#     }
#     delta_hat_commented <- delta_hat_commented + gamma_j_T
#   }
#
#   print("Delta (Current Code):")
#   print(delta_hat_current)
#
#   print("Delta (Commented-Out Code):")
#   print(delta_hat_commented)
#
#   ### Comparison ###
#   result <- all.equal(delta_hat_current, delta_hat_commented, tolerance = 1e-8)
#   if (isTRUE(result)) {
#     message("Both implementations yield the same result!")
#   } else {
#     message("The results differ:")
#     print(delta_hat_current)
#     print(delta_hat_commented)
#   }
# }
#
#
# Example Usage
# set.seed(123)
# Y <- sample(0:1, 100, replace = TRUE)
# D <- sample(0:1, 100, replace = TRUE)
# pi <- matrix(runif(100 * 11), nrow = 100, ncol = 11)
# nb_batch <- 10
# batch_indices <- split(1:100, rep(1:nb_batch, each = 10))
#
# validate_cram_estimator(Y, D, pi, batch_indices)
