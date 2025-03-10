# EXTRACT 2D FROm 3D ------------------------------------------------------------------------

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


# COMPUTE PROBA, to be applied on each agent, sim subgroup --------------------------------------------


# library(dplyr)
# library(tidyr)

# Function to compute probabilities for each row inside a group (agent, sim)
compute_probas <- function(df, policy, policy_name) {
  # Extract contexts for the entire (agent, sim) group (same for all t)
  contexts <- df$context  # Already a list
  ind_arm <- df$choice

  theta_all <- df[, theta]   # List-column containing A and b

  if (policy_name == "ContextualEpsilonGreedyPolicy") {
    # Extract matrices A and vectors b
    A_list <- lapply(theta_all, `[[`, "A")
    b_list <- lapply(theta_all, `[[`, "b")

    # Compute probabilities efficiently in batch
    probas_matrix <- get_proba_c_eps_greedy(policy$epsilon, A_list, b_list, contexts, ind_arm)  # Shape: (T × T)

  } else if (policy_name == "ContextualLinTSPolicy") {
    # Extract matrices A and vectors b
    A_list <- lapply(theta_all, `[[`, "A_inv")
    b_list <- lapply(theta_all, `[[`, "b")

    probas_matrix <- get_proba_thompson(policy$sigma, A_list, b_list, contexts, ind_arm)  # Shape: (T × T)

  } else if (policy_name == "LinUCBDisjointPolicyEpsilon") {

    # Extract matrices A and vectors b
    A_list <- lapply(theta_all, `[[`, "A")
    b_list <- lapply(theta_all, `[[`, "b")

    # Compute probabilities efficiently in batch
    probas_matrix <- get_proba_ucb_disjoint(policy$alpha, policy$epsilon, A_list, b_list, contexts, ind_arm)  # Shape: (T × T)

  } else {
    stop("Unsupported policy_name: Choose either 'epsilon-greedy' or 'thompson-sampling'")
  }

  # Store each COLUMN of probas_matrix in a list-column
  probas_list <- split(probas_matrix, col(probas_matrix))  # List of T vectors

  # Return df with the new probas column
  return(cbind(df, probas = probas_list))
}



# GET PROBA EPSILON GREEDY -------------------------------------------------------------------------

get_proba_c_eps_greedy <- function(eps = 0.1, A_list, b_list, contexts, ind_arm) {
  # A_list and b_list contain the list (across both sim and timesteps) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))  # Convert from list/data.table format if necessary
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # Get a list of length T where each element represents a policy:
  # a policy is K vectors theta = A^-1 b of resulting shape (d x 1), one per arm
  expected_rewards <- lapply(seq_len(nb_timesteps), function(t) {
    # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat
  })

  # Convert expected_rewards (list of T matrices) into a 3D array (T × K × T)
  # T x K x T = context x arm x policy
  expected_rewards_array <- simplify2array(expected_rewards)

  # Swap last dimension (T) with second dimension (K) → (T × T × K)
  # T x T x K = context x policy x arm
  expected_rewards_array <- aperm(expected_rewards_array, c(1, 3, 2))

  # Find max expected rewards for each row in every T × T matrix
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × T)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_timesteps, K))

  #   # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × T × K)


  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × T)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × T)

  # Compute final probabilities (T × T)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

  return(proba_results)  # Returns (T × T) matrix of probabilities, one context per row
}



# GET PROBA UCB DISJOINT WITH EPSILON ---------------------------------------------------------

get_proba_ucb_disjoint <- function(alpha=1.0, eps = 0.1, A_list, b_list, contexts, ind_arm) {
  # A_list and b_list contain the list (across both sim and timesteps) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))  # Convert from list/data.table format if necessary
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # Get a list of length T where each element represents a policy:
  # a policy is K vectors theta = A^-1 b of resulting shape (d x 1), one per arm
  mu <- lapply(seq_len(nb_timesteps), function(t) {
    # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat # (T x K)
  })

  variance <- lapply(seq_len(nb_timesteps), function(t) {
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% inv(A_list[[t]][[k]]) # (T x d)
      # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance_terms <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
      sqrt(variance_terms)
    }, simplify = "matrix") # (T x K)
  })

  # Convert expected_rewards (list of T matrices) into a 3D array (T × K × T)
  # T x K x T = context x arm x policy
  mu_array <- simplify2array(mu)
  variance_array <- simplify2array(variance)

  # Swap last dimension (T) with second dimension (K) → (T × T × K)
  # T x T x K = context x policy x arm
  mu_array <- aperm(mu_array, c(1, 3, 2))
  variance_array <- aperm(variance_array, c(1, 3, 2))

  expected_rewards_array <- mu_array + alpha * variance_array

  # Find max expected rewards for each row in every T × T matrix
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × T)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_timesteps, K))

  #   # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × T × K)


  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × T)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × T)

  # Compute final probabilities (T × T)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

}


# GET PROBA THOMPSON SAMPLING ---------------------------------------------------------------------


get_proba_thompson <- function(sigma = 0.01, A_list, b_list, contexts, ind_arm) {

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))  # Convert from list/data.table format if necessary
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # Get a list of length T where each element represents a policy:
  # a policy is K vectors theta = A^-1 b of resulting shape (d x 1), one per arm
  result <- lapply(seq_len(nb_timesteps), function(t) {

    # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
    theta_hat <- sapply(seq_len(K), function(k) A_list[[t]][[k]] %*% b_list[[t]][[k]], simplify = "matrix")

    # print("Shape theta_hat")
    # print(dim(theta_hat))

    mean <- context_matrix  %*% theta_hat # (T x K)
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% (sigma * A_list[[t]][[k]]) # (T x d)
      # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
    }, simplify = "matrix") # (T x K)

    # print("Shape mean")
    # print(dim(mean))
    #
    # print("Shape variance matrix")
    # print(dim(variance_matrix))

    proba_results <- numeric(nb_timesteps)

    # # Precompute competition indices once for all timesteps
    # comp_arms_list <- lapply(1:nb_timesteps, function(j) setdiff(1:K, ind_arm[j]))

    for (j in 1:nb_timesteps) {
      # Xa <- matrix(contexts[[t]], nrow = 1)  # Ensure Xa is a 1 × d matrix
      #
      # mean_k <- Xa %*% theta_hat[[ind_arm[t]]]
      # var_k  <-  Xa %*% sigma_hat[[ind_arm[t]]] %*% t(Xa)

      mean_k <- mean[j, ind_arm[j]]
      var_k  <-  variance_matrix[j, ind_arm[j]]

      competing_arms <- setdiff(1:K, ind_arm[j])
      # competing_arms <- comp_arms_list[[j]]

      # mean_values <- sapply(competing_arms, function(i) as.numeric(Xa %*% theta_hat[[i]]))
      # var_values  <- sapply(competing_arms, function(i) max(as.numeric(Xa %*% sigma_hat[[i]] %*% t(Xa)), 1e-6))  # Avoid zero variance

      mean_values <- mean[j,competing_arms]
      var_values <- variance_matrix[j, competing_arms]

      # print("Mean vals")
      # print(mean_values)
      #
      # print("Var vals")
      # print(var_values)
      #
      # print("competing arms")
      # print(competing_arms)


      # Define the function for integration
      integrand <- function(x) {
        # Compute the transformed mean and variance for the chosen arm

        log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

        for (i in seq_along(mean_values)) {
          log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[i], sd = sqrt(var_values[i]), log.p = TRUE)
        }

        return(exp(log_p_xk))  # Convert back to probability space
      }

      lower_bound <- mean_k - 3 * sqrt(var_k)
      upper_bound <- mean_k + 3 * sqrt(var_k)

      # Adaptive numerical integration
      prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 1000, rel.tol = 1e-2)$value
      # prob <- pracma::integral(integrand , -Inf , Inf )

      proba_results[j] <- pmax(0.05, pmin(prob, 0.95))
    }

    # print("proba results")
    # print(proba_results)

    return(proba_results)
  })

  # result is a list giving for each policy t, the array of probabilities under each context j
  # of selecting Aj
  result_matrix <- t(simplify2array(result)) # a row should be a context, policies in columns


  return(result_matrix)
}




# CUSTOM CONTEXTUAL LINEAR POLICIES -----------------------------------------------------------------


# UCB DISJOINT WITH EPSILON
LinUCBDisjointPolicyEpsilon <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    alpha = NULL,
    epsilon = NULL,
    class_name = "LinUCBDisjointPolicyEpsilon",
    initialize = function(alpha = 1.0, epsilon=0.1) {
      super$initialize()
      self$alpha <- alpha
      self$epsilon <- epsilon
    },
    set_parameters = function(context_params) {
      ul <- length(context_params$unique)
      self$theta_to_arms <- list('A' = diag(1,ul,ul), 'b' = rep(0,ul))
    },
    get_action = function(t, context) {

      if (runif(1) > self$epsilon) {

        expected_rewards <- rep(0.0, context$k)

        for (arm in 1:context$k) {

          Xa         <- get_arm_context(context, arm, context$unique)
          A          <- self$theta$A[[arm]]
          b          <- self$theta$b[[arm]]

          A_inv      <- inv(A)

          theta_hat  <- A_inv %*% b

          mu_hat     <- Xa %*% theta_hat
          sigma_hat  <- sqrt(tcrossprod(Xa %*% A_inv, Xa))

          expected_rewards[arm] <- mu_hat + self$alpha * sigma_hat
        }
        action$choice  <- which_max_tied(expected_rewards)

      } else {

        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)
      }

      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa     <- get_arm_context(context, arm, context$unique)

      inc(self$theta$A[[arm]]) <- outer(Xa, Xa)
      inc(self$theta$b[[arm]]) <- reward * Xa

      self$theta
    }
  )
)
