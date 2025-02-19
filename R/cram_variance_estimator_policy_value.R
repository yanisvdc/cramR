#' Cram Variance Estimator for Policy Value (Psi)
#'
#' This function estimates the asymptotic variance of the cram estimator
#' for the policy value (psi).
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns,
#'           where n is the sample size and nb_batch is the number of batches,
#'           containing the policy assignment for each individual for each policy.
#'           The first column represents the baseline policy.
#' @param batch_indices A list where each element is a vector of indices corresponding to the individuals in each batch.
#' @param propensity Propensity score function
#' @return The estimated variance \eqn{\hat{w}^2_T}.
#' @examples
#' # Example usage:
#' Y <- sample(0:1, 100, replace = TRUE)
#' D <- sample(0:1, 100, replace = TRUE)
#' pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
#' nb_batch <- 10
#' batch_indices <- split(1:100, rep(1:nb_batch, each = 10))
#' variance_estimate <- cram_variance_estimator_policy_value(Y, D, pi, batch_indices)
#' @export
cram_variance_estimator_policy_value <- function(X, Y, D, pi, batch_indices, propensity = NULL) {
  # Determine number of batches
  nb_batch <- length(batch_indices)
  # Batch size (assuming all batches are the same size)
  batch_size <- length(batch_indices[[length(batch_indices)]])

  if (nrow(pi) != length(Y) || length(Y) != length(D)) {
    stop("Y, D, and pi must have matching lengths")
  }

  if (is.null(propensity)) {
    # IPW component for all individuals using default propensity = 0.5
    weight_diff <- Y * D / 0.5 - Y * (1 - D) / 0.5
    part_1_weight <- Y * D / 0.5
    part_2_weight <- Y * (1 - D) / 0.5
  } else {
    # Compute propensity scores
    propensity_scores <- propensity(X)

    # Compute IPW component with custom propensity scores
    weight_diff <- Y * D / propensity_scores - Y * (1 - D) / propensity_scores
    part_1_weight <- Y * D / propensity_scores
    part_2_weight <- Y * (1 - D) / propensity_scores
  }

  policy_diff <- pi[, 2:nb_batch] - pi[, 1:(nb_batch - 1)]

  # Remove first column
  policy_diff <- policy_diff[, -1]

  # Vector of terms for each column
  policy_diff_weights <- 1 / (nb_batch - (2:(nb_batch - 1)))

  # Multiply each column of policy_diff by corresponding policy_diff_weight
  policy_diff <- sweep(policy_diff, 2, policy_diff_weights, FUN = "*")

  policy_diff <- t(apply(policy_diff, 1, cumsum))

  policy_diff <- sweep(policy_diff, 1, weight_diff, FUN = "*")

  first_col <- (part_1_weight * pi[, 2] + part_2_weight * (1-pi[, 2])) / (nb_batch - 1)

  policy_diff <- cbind(first_col, policy_diff)

  # Create the mask for batch indices
  mask <- matrix(1, nrow = nrow(policy_diff), ncol = ncol(policy_diff))

  for (k in seq_len(nb_batch - 1)) {
    # Set to NA the rows corresponding to batches 1 to k in column k
    mask[unlist(batch_indices[1:k]), k] <- NA
  }

  # Apply the mask to policy_diff
  policy_diff <- policy_diff * mask

  # Calculate variance for each column
  column_variances <- apply(policy_diff, 2, function(x) var(x, na.rm = TRUE))

  total_variance <- sum(column_variances)

  # Final variance estimator, scaled by T / B
  total_variance <- (nb_batch / batch_size) * total_variance

  return(total_variance)
}
