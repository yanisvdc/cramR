#' Cram Policy: Variance Estimate of the crammed Policy Value Difference (Delta)
#'
#' This function estimates the asymptotic variance of the cram estimator
#' for the policy value difference (delta).
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
#' @return The estimated variance of the policy value difference (Delta)
#' @export
cram_variance_estimator <- function(X, Y, D, pi, batch_indices, propensity = NULL) {
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
  mask <- matrix(1, nrow = nrow(policy_diff), ncol = ncol(policy_diff))

  for (k in seq_len(nb_batch - 1)) {
    # Set to NA the rows corresponding to batches 1 to k in column k
    mask[unlist(batch_indices[1:k]), k] <- NA
  }

  # Apply the mask to policy_diff
  policy_diff <- policy_diff * mask

  policy_diff <- sweep(policy_diff, 1, weight_diff, FUN = "*")

  # Calculate variance for each column
  column_variances <- apply(policy_diff, 2, function(x) var(x, na.rm = TRUE))

  total_variance <- sum(column_variances)

  total_variance <- (1 / batch_size) * total_variance

  return(total_variance)
}
