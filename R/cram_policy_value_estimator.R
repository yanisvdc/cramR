#' Cram Policy: Estimator for Policy Value (Psi)
#'
#' This function returns the cram estimator for the policy value (psi).
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
#' @return The estimated policy value.
#' @export

cram_policy_value_estimator <- function(X, Y, D, pi, batch_indices, propensity = NULL) {

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

  # At this point, we have nb_batch - 1 columns ---------------------
  # The columns contain Gamma_j(T) for j from 2 to nb_batch
  # Column k contains Gamma_{k+1}(T), for k from 1 to nb_batch - 1
  # For each column k, only the rows for individuals in batch {k+1} are not NA

  # How to adjust for Psi calculation:
  ## Step 1: calculate the terms from eta_j for j=2, .., T (inside the average)

  # part_1_weight <- Y * D / 0.5
  #
  # part_2_weight <- Y * (1 - D) / 0.5

  eta_terms <- (part_1_weight * pi[, 1] + part_2_weight * (1-pi[, 1])) / nb_batch

  ## Step 2: at this point we have all the columns to average for j = 2, ..., T
  # We need to add the column for j = 1, to get Psi_1(T) when averaged

  policy_diff <- cbind(NA, policy_diff)

  batch_1_indices <- batch_indices[[1]]  # Get the indices for batch 1

  policy_diff[batch_1_indices, 1] <- 0

  ## Final step: now it suffices to add the eta_terms vector to all columns
  # It will preserve the NA and only add to the relevant batches

  policy_diff <- policy_diff + eta_terms

  # Calculate average for each column
  column_averages <- apply(policy_diff, 2, function(x) mean(x, na.rm = TRUE))

  psi_hat <- sum(column_averages)

  return(psi_hat)
}
