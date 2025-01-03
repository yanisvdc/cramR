#' Crammed Variance Estimator for Policy Evaluation
#'
#' This function estimates the variance of the cram estimator to measure the asymptotic
#' variance of policy value differences.
#'
#' @param Y A vector of outcomes for the n individuals.
#' @param D A vector of binary treatments for the n individuals.
#' @param pi A matrix of n rows and (nb_batch + 1) columns, containing the policy assignment for each individual
#'           for each policy. The first column represents the baseline policy.
#' @param batch_indices A list where each element is a vector of indices corresponding to the individuals in each batch.
#' @return The estimated variance \eqn{\hat{v}^2_T}.
#' @examples
#' # Example usage:
#' Y <- sample(0:1, 100, replace = TRUE)
#' D <- sample(0:1, 100, replace = TRUE)
#' pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
#' nb_batch <- 10
#' batch_indices <- split(1:100, rep(1:nb_batch, each = 10))
#' variance_estimate <- cram_variance_estimator(Y, D, pi, batch_indices)
#' @export
cram_variance_estimator <- function(Y, D, pi, batch_indices) {
  # Determine number of batches
  nb_batch <- length(batch_indices)
  # Batch size (assuming all batches are the same size)
  batch_size <- length(batch_indices[[length(batch_indices)]])

  if (nrow(pi) != length(Y) || length(Y) != length(D)) {
    stop("Y, D, and pi must have matching lengths")
  }

  # IPW component for all individuals
  weight_diff <- Y * D / 0.5 - Y * (1 - D) / 0.5

  policy_diff <- pi[, 2:nb_batch] - pi[, 1:(nb_batch - 1)]

  # Vector of terms for each column
  policy_diff_weights <- 1 / (nb_batch - (1:(nb_batch - 1)))

  # Multiply each column of policy_diff by corresponding policy_diff_weight
  policy_diff <- sweep(policy_diff, 2, policy_diff_weights, FUN = "*")

  policy_diff <- t(apply(policy_diff, 1, cumsum))

  # Create the mask for batch indices
  mask <- matrix(1, nrow = nrow(policy_diff), ncol = ncol(policy_diff))

  for (k in seq_len(nb_batch - 1)) {
    # Set to 0 the rows corresponding to batches 1 to k in column k
    mask[unlist(batch_indices[1:k]), k] <- NA
  }

  # Apply the mask to policy_diff
  policy_diff <- policy_diff * mask

  policy_diff <- sweep(policy_diff, 1, weight_diff, FUN = "*")

  # Calculate variance for each column
  column_variances <- apply(policy_diff, 2, function(x) var(x, na.rm = TRUE))

  total_variance <- sum(column_variances)

  # Final variance estimator, scaled by T / B
  total_variance <- (nb_batch / batch_size) * total_variance


  # total_length <- length(unlist(batch_indices))

  # # Loop through each batch (from j = 2 to T)
  # for (j in 2:nb_batch) {
  #   indices <- unlist(batch_indices[j:nb_batch])
  #
  #   # Collect all g_hat_Tj values across batches k = j to T
  #   # Preallocate g_hat_Tj_values with the correct length
  #   # total_length <- total_length - length(batch_indices[[j-1]])
  #   # g_hat_Tj_values <- numeric(total_length)
  #
  #   if (j == 2) {
  #     # For j == 2, the difference is simpler
  #     # just one element here
  #     vector_summed_differences <- (pi[indices, 2] - pi[indices, 1]) / (nb_batch - 1)
  #   } else {
  #     # Compute row-wise weighted differences efficiently
  #     policy_matrix_diffs <- pi[indices, 2:j] - pi[indices, 1:(j - 1)]
  #     diffs_denominator <- 1 / (nb_batch - (1:(j - 1)))
  #     vector_summed_differences <- policy_matrix_diffs %*% diffs_denominator
  #   }
  #   g_hat_Tj_values <- weight_diff[indices] * vector_summed_differences

    # pi[indices, 2:j] - pi[indices, 1:(j - 1)]
    #
    # for (k in j:nb_batch) {
    #   # Compute g_hat_Tj for each individual in batch k
    #   indices <- batch_indices[[k]]
    #   if (j==2){
    #     # Directly calculate the difference for j == 2
    #     policy_diff_sum <- (pi[indices, 2] - pi[indices, 1]) / (nb_batch - 1)
    #   } else {
    #     # Compute the policy difference sum (vectorized) for batch k
    #     # Column t is difference t - t-1, divided by T-t, we then sum per row
    #     policy_diff_sum <- rowSums((pi[indices, 2:j] - pi[indices, 1:(j - 1)]) / (nb_batch - (1:(j - 1))))
    #   }
    #   # Calculate g_hat_Tj for individuals in batch k
    #   g_hat_Tj_batch <- weight_diff[indices] * policy_diff_sum
    #   g_hat_Tj_values <- c(g_hat_Tj_values, g_hat_Tj_batch)
    # }

    # # Mean of g_hat_Tj over the batches from j to T
    # g_bar_Tj <- mean(g_hat_Tj_values)
    #
    # # Compute V_hat(g_hat_Tj) for batch j
    # if (batch_size == 1 && j == nb_batch) {
    #   V_hat_g_Tj <- 0  # Set variance to zero if batch size is one and j = T
    # } else {
    #   V_hat_g_Tj <- sum((g_hat_Tj_values - g_bar_Tj)^2) / (length(g_hat_Tj_values) - 1)
    # }

  #   V_hat_g_Tj <- var(g_hat_Tj_values)
  #
  #   # Add contribution of this batch to the total variance estimator
  #   variance_hat <- variance_hat + V_hat_g_Tj
  # }
  #
  # # Final variance estimator, scaled by T / B
  # variance_hat <- (nb_batch / batch_size) * variance_hat

  return(total_variance)
}


# Y <- sample(0:1, 100, replace = TRUE)
# D <- sample(0:1, 100, replace = TRUE)
# pi <- matrix(sample(0:1, 100 * 11, replace = TRUE), nrow = 100, ncol = 11)
# variance_estimate <- cram_variance_estimator(Y, D, pi, batch_indices)
# print(variance_estimate)
