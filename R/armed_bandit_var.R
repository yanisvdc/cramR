#' Cram Bandit Variance
#'
#' This function implements the crammed variance estimator for bandit policy
#'
#' @param pi A 3-d array, for each row j, column t, depth a, gives pi_t(Xj, a)
#' @param reward A vector of rewards
#' @param arm A vector of arms chosen
#' @return The crammed variance estimate for bandit.
#' @export

cram_bandit_var <- function(pi, reward, arm) {
  dims_result <- dim(pi)

  if (length(dims_result) == 3) {

    # Extract relevant dimensions
    nb_arms <- dims_result[3]
    nb_timesteps <- dims_result[2]


    ## POLICY SLICED: remove the arm dimension as Xj is associated to Aj

    # pi:
    # for each row j, column t, depth a, gives pi_t(Xj, a)

    # We do not need the last column and the first row
    # We only need, for each row j, pi_t(Xj, Aj), where Aj is the arm chosen from context j
    # Aj is the jth element of the vector arm, and corresponds to a depth index

    # drop = False to maintain 3D structure

    pi <- pi[-1, -ncol(pi), , drop = FALSE]

    depth_indices <- arm[2:nb_timesteps]

    pi <- extract_2d_from_3d(pi, depth_indices)

  } else {

    nb_timesteps <- dims_result[2]

    pi <- pi[-1, -ncol(pi), drop = FALSE]

  }

  # pi is now a (T-1) x (T-1) matrix

  ## POLICY DIFFERENCE

  # Before doing policy differences with lag 1, we add a column of 0
  # such that the first policy difference corresponds to pi_1
  pi <- cbind(0, pi)

  pi_diff <- pi[, -1] - pi[, -ncol(pi)]

  # Vector of terms for each column
  policy_diff_weights <- 1 / (nb_timesteps - (1:(nb_timesteps - 1)))

  # Multiply each column of pi_diff by corresponding policy_diff_weight
  pi_diff <- sweep(pi_diff, 2, policy_diff_weights, FUN = "*")

  # Cumulative sum
  pi_diff <- t(apply(pi_diff, 1, cumsum))

  # pi_diff is a (T-1) x (T-1) matrix


  ## MULTIPLY by Rk / pi_k-1

  # Get diagonal elements from pi
  pi_diag <- pi[cbind(1:(nrow(pi)), 2:(ncol(pi)))]

  # Create multipliers using vectorized operations
  multipliers <- (1 / pi_diag) * reward[2:length(reward)]

  # Apply row-wise multiplication using efficient matrix operation
  # mult_pi_diff <- pi_diff * multipliers  # Works via R's recycling rules (most efficient)
  mult_pi_diff <- sweep(pi_diff, 1, multipliers, FUN = "*")


  ## VARIANCE PER COLUMN

  # Create the mask for batch indices
  mask <- matrix(1, nrow = nrow(mult_pi_diff), ncol = ncol(mult_pi_diff))

  # Assign NaN to upper triangle elements (excluding diagonal)
  mask[upper.tri(mask, diag = FALSE)] <- NaN

  # Apply the mask to policy_diff
  mult_pi_diff <- mult_pi_diff * mask

  # Calculate variance for each column
  # column_variances <- apply(mult_pi_diff, 2, function(x) var(x, na.rm = TRUE))
  # column_variances <- apply(mult_pi_diff, 2, function(x) {
  #   n <- sum(!is.na(x))  # Count of non-NA values
  #   sample_var <- var(x, na.rm = TRUE)  # Compute sample variance (divides by n-1)
  #   sample_var * (n - 1) / n  # Convert to population variance (divides by n)
  # })

  column_variances <- apply(mult_pi_diff, 2, function(x) {
    n <- sum(!is.na(x))  # Count of non-NA values
    if (n == 1){
      return(0)
    } else {
      sample_var <- var(x, na.rm = TRUE)  # Compute sample variance (divides by n-1)
      sample_var <- sample_var * (n - 1) / n  # Convert to population variance (divides by n)
      return(sample_var)
    }
  })

  # print(column_variances)
  total_variance <- sum(column_variances)

  return(total_variance)
}

