#' Variance Estimation for Cram Estimator
#'
#' This function computes the variance estimator \eqn{\hat{\sigma}^2_2}
#' based on the given loss matrix and batch indices.
#'
#' @param loss A matrix of loss values with \eqn{N} rows (data points) and \eqn{K} columns (batches).
#' @param batch_indices A list where each element is a vector of indices corresponding to a batch.
#' @return The estimated variance \eqn{\hat{\sigma}^2_2}.
#' @examples
#' # Example usage
#' set.seed(123)
#' N <- 100  # Number of data points
#' K <- 10   # Number of batches
#'
#' loss <- matrix(rnorm(N * K), nrow = N, ncol = K)  # Random loss values
#' batch_indices <- split(1:N, rep(1:K, each = N / K))
#'
#' cram_variance_estimation(loss, batch_indices)
#' @export

cram_variance_estimation <- function(loss, batch_indices) {
  # Check inputs
  if (!is.matrix(loss)) {
    stop("`loss` must be a matrix with N rows (data points) and K columns (batches).")
  }
  if (!is.list(batch_indices)) {
    stop("`batch_indices` must be a list of batch index vectors.")
  }

  N <- nrow(loss)  # Number of data points
  K <- ncol(loss)  # Number of batches

  # Initialize variance estimate
  variance_estimate <- 0

  # Loop over batches
  for (l in 2:K) {
    # Combined batch indices from l to K
    combined_indices <- unlist(batch_indices[l:K])

    # Compute the inner summation for each i
    inner_sums <- sapply(combined_indices, function(i) {
      sum(sapply(1:(l - 1), function(k) {
        (loss[i, k + 1] - loss[i, k]) / (K - k)
      }))
    })

    # Compute the sample variance of the inner sums
    sample_variance <- var(inner_sums)

    # Accumulate the variance estimate
    variance_estimate <- variance_estimate + sample_variance
  }

  # Multiply by K to finalize the variance estimate
  variance_estimate <- K * variance_estimate

  return(variance_estimate)
}
