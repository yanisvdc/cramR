#' Cram ML Expected Loss Estimate
#'
#' This function computes the Cram ML expected loss estimator
#' based on the given loss matrix and batch indices.
#'
#' @param loss A matrix of loss values with N rows (data points) and K+1 columns (batches).
#' We assume that the first column of the loss matrix contains only zeros.
#' The following nb_batch columns contain the losses of each trained model for each individual.
#' @param batch_indices A list where each element is a vector of indices corresponding to a batch.
#' @return The Cram ML expected loss estimate
#' @export

cram_expected_loss <- function(loss, batch_indices) {
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

  # Create the mask for batch indices
  mask <- matrix(NA, nrow = nrow(loss_diff), ncol = ncol(loss_diff))

  for (k in 2:nb_batch) {
    # Set to NA the rows corresponding to batches k and above for column k-1
    mask[unlist(batch_indices[k:nb_batch]), k-1] <- 1
  }

  # Apply the mask to loss_diff
  loss_diff <- loss_diff * mask

  # Calculate average for each column
  column_averages <- apply(loss_diff, 2, function(x) mean(x, na.rm = TRUE))

  cram_expected_loss <- sum(column_averages)

  return(cram_expected_loss)
}
