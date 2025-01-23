#' CRAM Expected Loss Estimation
#'
#' This function computes the Cram expected loss estimator \eqn{\hat{R}_{\mathrm{cram}}}
#' based on the given loss matrix and batch indices.
#'
#' @param loss A matrix of loss values with \eqn{N} rows (data points) and \eqn{K} columns (batches).
#' @param batch_indices A list where each element is a vector of indices corresponding to a batch.
#'                      For example: \code{split(1:N, rep(1:nb_batch, each = N / nb_batch))}.
#' @return The estimated Cram expected loss \eqn{\hat{R}_{\mathrm{cram}}}.
#' @examples
#' # Example usage
#' set.seed(123)
#' N <- 100  # Number of data points
#' K <- 10   # Number of batches
#'
#' loss <- matrix(rnorm(N * K), nrow = N, ncol = K)  # Random loss values
#' batch_indices <- split(1:N, rep(1:K, each = N / K))
#'
#' cram_expected_loss(loss, batch_indices)
#' @export

cram_expected_loss <- function(loss, batch_indices) {
  # Check inputs
  if (!is.matrix(loss)) {
    stop("`loss` must be a matrix with N rows (data points) and K columns (batches).")
  }
  if (!is.list(batch_indices)) {
    stop("`batch_indices` must be a list of batch index vectors.")
  }

  N <- nrow(loss)  # Number of data points
  K <- ncol(loss)  # Number of batches
  B <- length(batch_indices[[1]])  # Batch size

  # Initialize the Cram expected loss
  R_cram <- 0

  # Loop over batches
  for (k in 1:(K - 1)) {
    # Retrieve the batch indices for evaluation
    current_batch <- unlist(batch_indices[(k + 1):K])  # Indices for \( i \geq kB + 1 \)

    # Compute the cumulative loss difference for the batch
    loss_diff <- sum(loss[current_batch, k + 1] - loss[current_batch, k])

    # Update the Cram expected loss
    R_cram <- R_cram + (1 / ((K - k) * B)) * loss_diff
  }

  return(R_cram)
}
