#' Variance Estimation for Cram Estimator
#'
#' This function computes the variance estimator \eqn{\hat{\sigma}^2_2}
#' based on the given loss matrix and batch indices.
#'
#' @param loss A matrix of loss values with \eqn{N} rows (data points) and \eqn{K+1} columns (batches).
#' We assume that the first column of the loss matrix contains only zeros.
#' The following nb_batch columns contain the losses of each trained model for each individual.
#' @param batch_indices A list where each element is a vector of indices corresponding to a batch.
#'                      For example: \code{split(1:N, rep(1:nb_batch, each = N / nb_batch))}.
#' @return The estimated Cram expected loss \eqn{\hat{R}_{\mathrm{cram}}}.
#' @examples
#' # Example usage
#' set.seed(123)
#' N <- 100  # Number of data points
#' K <- 10   # Number of batches
#'
#' # Generate a loss matrix with K+1 columns, first column as zeros
#' loss <- matrix(rnorm(N * (K+1)), nrow = N, ncol = K+1)
#' loss[, 1] <- 0  # Ensure first column is zero
#'
#' # Create batch indices dynamically
#' batch_indices <- split(1:N, rep(1:K, length.out = N))
#'
#' # Compute Cram Expected Loss Variance
#' cram_var_expected_loss(loss, batch_indices)
#' @export

cram_var_expected_loss <- function(loss, batch_indices) {
  # Check inputs
  if (!is.matrix(loss)) {
    stop("`loss` must be a matrix with N rows (data points) and K+1 columns (batches).")
  }
  if (!is.list(batch_indices)) {
    stop("`batch_indices` must be a list of batch index vectors.")
  }

  N <- nrow(loss)  # Number of data points
  nb_batch <- length(batch_indices)  # Number of batches

  loss_diff <- loss[, 2:nb_batch] - loss[, 1:(nb_batch - 1)]

  # Vector of terms for each column
  loss_diff_weights <- 1 / (nb_batch - (1:(nb_batch - 1)))

  # Multiply each column of loss_diff by corresponding loss_diff_weight
  loss_diff <- sweep(loss_diff, 2, loss_diff_weights, FUN = "*")

  loss_diff <- t(apply(loss_diff, 1, cumsum))

  # Create the mask for batch indices
  mask <- matrix(NA, nrow = nrow(loss_diff), ncol = ncol(loss_diff))

  for (k in 2:nb_batch) {
    # Set to NA the rows corresponding to batches k and above for column k-1
    mask[unlist(batch_indices[k:nb_batch]), k-1] <- 1
  }

  # Apply the mask to loss_diff
  loss_diff <- loss_diff * mask

  # Calculate variance for each column
  column_variances <- apply(loss_diff, 2, function(x) var(x, na.rm = TRUE))

  total_variance <- sum(column_variances)

  # Final variance estimator, scaled by T
  total_variance <- nb_batch * total_variance

  return(total_variance)
}
