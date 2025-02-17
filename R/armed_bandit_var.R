#' Crammed Variance Estimator
#'
#' This function implements the crammed variance estimator \eqn{\hat{\sigma}^2_T}
#' as described in the provided formula.
#'
#' @param Y A vector of observed rewards for each time step.
#' @param D A vector of treatment indicators (1 if treated, 0 if not) for each time step.
#' @param pi A matrix where each row represents the policy probabilities for actions at time t.
#' @param X A matrix or data frame of covariates observed at each time step.
#' @param T The total number of iterations in the bandit process.
#' @return The crammed variance estimate \eqn{\hat{\sigma}^2_T}.
#' @export

cram_bandit_var <- function(pi, reward, arm) {
  if (length(Y) != T || length(D) != T || nrow(pi) != T || nrow(X) != T) {
    stop("All input vectors/matrices must have the same number of rows equal to T.")
  }

  # Initialize the overall variance estimate
  sigma_squared_T <- 0

  for (j in 2:T) {
    # Initialize the variance estimate for each j
    variance_Tj <- 0

    # Calculate the GTjk values for k = j to T
    GTjk <- numeric(T - j + 1)
    for (k in j:T) {
      term1 <- (Y[k] * D[k]) / pi[k - 1, D[k] + 1]
      term2 <- (Y[k] * (1 - D[k])) / (1 - pi[k - 1, D[k] + 1])
      inner_sum <- 0

      for (a in 1:ncol(pi)) {
        for (t in 1:(j - 1)) {
          weight <- (pi[t, a] - pi[t - 1, a]) / (T - t)
          inner_sum <- inner_sum + weight
        }
      }
      GTjk[k - j + 1] <- (term1 - term2) * inner_sum
    }

    # Compute the mean of GTjk
    GTj_mean <- mean(GTjk)

    # Compute the variance for j
    variance_Tj <- sum((GTjk - GTj_mean)^2) / (T - j)

    # Add to the overall variance estimate
    sigma_squared_T <- sigma_squared_T + variance_Tj
  }

  # Scale by T as per the formula
  sigma_squared_T <- T * sigma_squared_T

  return(sigma_squared_T)
}
