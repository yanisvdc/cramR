extract_2d_from_3d <- function(array3d, depth_indices) {
  # Get array dimensions
  dims <- dim(array3d)
  nrow <- dims[1]  # Rows
  ncol <- dims[2]  # Columns

  # Ensure depth_indices length matches required rows
  if (length(depth_indices) != nrow) {
    stop("The arm selection vector should have same length as the first dimension of the policy array.")
  }

  # Vectorized index calculation
  i <- rep(1:nrow, each = ncol)  # Row indices
  j <- rep(1:ncol, times = nrow) # Column indices
  k <- rep(depth_indices, each = ncol)  # Depth indices

  # Calculate linear indices for efficient extraction
  linear_indices <- i + (j - 1) * nrow + (k - 1) * nrow * ncol

  # Create result matrix using vectorized indexing
  result_matrix <- matrix(array3d[linear_indices], nrow = nrow, ncol = ncol, byrow = TRUE)

  return(result_matrix)
}

# # Define parameters
# T <- 5   # Total time points
# K <- 2   # Number of depth layers
#
# # Create a 3D array (T-2 x T-1 x K)
# array3d <- array(1:((T-2)*(T-1)*K), dim = c(T-2, T-1, K))
#
# # Create arm vector (must contain indices between 1 and K)
# arm <- c(1, 2, 1, 2, 1)  # Example arm vector
#
# # Vectorized index calculation
# depth_indices <- arm[3:T]  # Select arm[i+2] values
#
# res <- extract_2d_from_3d(array3d, depth_indices)







# # Define new test parameters
# T_new <- 6   # New total time points
# K_new <- 3   # New number of depth layers
#
# # Create a new 3D array (T-2 x T-1 x K)
# array3d_new <- array(1:((T_new-2)*(T_new-1)*K_new), dim = c(T_new-2, T_new-1, K_new))
#
# # Create a new arm vector (must contain indices between 1 and K)
# arm_new <- c(2, 3, 1, 2, 3, 1)  # Example arm vector
#
# # Extract depth indices from arm
# depth_indices_new <- arm_new[3:T_new]  # Select arm[i+2] values
#
# # Apply function to new data
# result_new <- extract_2d_from_3d(array3d_new, depth_indices_new)
#
# # Output results
# cat("\nNew 3D Array:\n")
# print(array3d_new)
#
# cat("\nNew Arm Vector:\n")
# print(arm_new)
#
# cat("\nNew Depth Indices:\n")
# print(depth_indices_new)
#
# cat("\nExtracted 2D Matrix:\n")
# print(result_new)
#



get_proba_c_eps_greedy <- function(eps = 0.1, A, b, contexts, ind_arm) {
  # ind_arm is the vector of indices of the arms that was chosen at each t

  K <- length(b)  # Number of arms
  nb_timesteps <- length(contexts)

  proba_results <- numeric(nb_timesteps)
  theta_hat <- vector("list", K)

  # Compute theta_hat for each arm efficiently
  for (arm in 1:K) {
    theta_hat[[arm]] <- solve(A[[arm]], b[[arm]])  # Avoid explicit inversion
  }

  # Compute expected rewards efficiently
  for (t in 1:nb_timesteps) {
    Xa <- contexts[[t]]

    # Vectorized calculation
    expected_rewards <- drop(Xa %*% do.call(cbind, theta_hat))  # Ensuring it's a vector

    # Find the indices of the best arms (ties)
    ties <- which(expected_rewards == max(expected_rewards))  # Get indices of max expected rewards

    # Compute probability
    if (ind_arm[t] %in% ties) {
      proba_results[t] <- (1 - eps) / length(ties) + eps / K
    } else {
      proba_results[t] <- eps / K
    }
  }

  return(proba_results)
}


get_proba_thompson <- function(sigma = 0.01, A_inv, b, contexts, ind_arm) {
  # ind_arm is the index of the arm that was chosen

  K <- length(b)  # Number of arms
  nb_timesteps <- length(contexts)

  proba_results <- numeric(nb_timesteps)
  theta_hat <- vector("list", K)
  sigma_hat <- vector("list", K)

  # Compute theta_hat for each arm efficiently
  for (arm in 1:K) {
    theta_hat[[arm]] <- A_inv[[arm]] %*% b[[arm]]
    sigma_hat[[arm]] <- sigma * A_inv[[arm]]
  }

  # Compute expected rewards efficiently
  for (t in 1:nb_timesteps) {
    Xa <- matrix(contexts[[t]], nrow = 1)  # Ensure Xa is a 1 × d matrix

    mean_k <- Xa %*% theta_hat[[ind_arm]]
    var_k  <-  Xa %*% sigma_hat[[ind_arm]] %*% t(Xa)

    competing_arms <- setdiff(1:K, ind_arm)

    mean_values <- sapply(competing_arms, function(i) as.numeric(Xa %*% theta_hat[[i]]))
    var_values  <- sapply(competing_arms, function(i) max(as.numeric(Xa %*% sigma_hat[[i]] %*% t(Xa)), 1e-6))  # Avoid zero variance

    # Define the function for integration
    integrand <- function(x) {
      # Compute the transformed mean and variance for the chosen arm

      log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

      for (j in seq_along(mean_values)) {
        log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[j], sd = sqrt(var_values[j]), log.p = TRUE)
      }

      # max_log_cdf <- apply(log_cdf_values, 1, function(row) max(row))  # (15 × 1) row-wise max
      # log_cdf_sum <- max_log_cdf + log(rowSums(exp(log_cdf_values - max_log_cdf)))  # (15 × 1)
      # log_cdf_sum <- rowSums(log_cdf_values)

      # # Compute log-probabilities that all other arms have a lower reward
      # log_cdf_sum <- sum(sapply(setdiff(1:K, ind_arm), function(i) {
      #   mean_i <- Xa %*% theta_hat[[i]]
      #   var_i  <- Xa %*% sigma_hat[[i]] %*% Xa
      #   pnorm(x, mean = mean_i, sd = sqrt(var_i), log.p = TRUE)  # Log-CDF
      # }))
      # Compute log-probabilities for all other arms
      # log_cdf_values <- sapply(setdiff(1:K, ind_arm), function(i) {
      #   mean_i <- Xa %*% theta_hat[[i]]
      #   var_i  <- as.numeric(Xa %*% sigma_hat[[i]] %*% t(Xa))  # Convert to scalar
      #   pnorm(x, mean = mean_i, sd = sqrt(var_i), log.p = TRUE)  # Log-CDF
      # })
      #
      # log_cdf_sum <- sum(log_cdf_values)  # Sum logs before exponentiation
      #

      return(exp(log_p_xk))  # Convert back to probability space
    }

    # lower_bound <- mean_k - 10 * sqrt(var_k)
    # upper_bound <- mean_k + 10 * sqrt(var_k)

    # Adaptive numerical integration
    # prob <- pracma::integral(integrand, lower_bound, upper_bound)
    prob <- integrate(integrand, lower = -Inf, upper = Inf, subdivisions = 500, rel.tol = 1e-2)$value

    proba_results[t] <- prob
  }
  return(proba_results)
}

library(dplyr)
library(tidyr)

# Function to compute probabilities for each row inside a group (agent, sim)
compute_probas <- function(df) {
  # Extract contexts for the entire (agent, sim) group (same for all t)
  contexts <- df$context  # Already a list
  ind_arm <- df$choice

  # Initialize a list to store probabilities for each row
  probas <- vector("list", nrow(df))

  # Iterate over each time step `t` within this (agent, sim) group
  for (i in seq_len(nrow(df))) {
    # Extract A and b for the specific time step `t`
    A <- df$theta[[i]]$A  # Already a list of matrices
    b <- df$theta[[i]]$b  # Already a list of vectors

    # Extract chosen arm for this row (ind_arm is just the 'choice' column)

    # Compute probability using the function
    probas[[i]] <- get_proba_c_eps_greedy(eps = 0.1, A = A, b = b, contexts = contexts, ind_arm = ind_arm)
    # probas[[i]] <- get_proba_thompson(A_inv = A_inv, b = b, contexts = contexts, ind_arm = ind_arm)
  }

  # Add probabilities back to the dataframe as a list-column
  df$probas <- probas
  return(df)
}
