#' @importFrom dplyr filter
#' @importFrom magrittr %>%

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

compute_probas <- function(df, policy, policy_name, batch_size) {
  # Extract contexts and arms for the entire (agent, sim) group
  contexts <- df$context
  ind_arm <- df$choice
  theta_all <- df[, theta]

  # Use A_inv if LinTSPolicy is in the string of policy_name
  key_A <- ifelse(grepl("LinTSPolicy", policy_name), "A_inv", "A")
  key_b <- "b"

  # Extract A and b
  A_list <- lapply(theta_all, `[[`, key_A)
  b_list <- lapply(theta_all, `[[`, key_b)

  # Subset A and b for batch processing
  if (batch_size > 1) {
    indices_to_keep <- seq(batch_size, length(A_list), by = batch_size)
    A_list <- A_list[indices_to_keep]
    b_list <- b_list[indices_to_keep]
  }

  # Compute the probability matrix based on the policy name
  probas_matrix <- switch(
    policy_name,
    "ContextualEpsilonGreedyPolicy" =,
    "BatchContextualEpsilonGreedyPolicy" = get_proba_c_eps_greedy(policy$epsilon, A_list, b_list, contexts, ind_arm, batch_size),

    "ContextualLinTSPolicy" =,
    "BatchContextualLinTSPolicy" = get_proba_thompson(policy$sigma, A_list, b_list, contexts, ind_arm, batch_size),

    "LinUCBDisjointPolicyEpsilon" =,
    "BatchLinUCBDisjointPolicyEpsilon" = get_proba_ucb_disjoint(policy$alpha, policy$epsilon, A_list, b_list, contexts, ind_arm, batch_size),

    stop("Unsupported policy_name: Choose among ContextualEpsilonGreedyPolicy, BatchContextualEpsilonGreedyPolicy,
         ContextualLinTSPolicy, BatchContextualLinTSPolicy, LinUCBDisjointPolicyEpsilon, BatchLinUCBDisjointPolicyEpsilon")
  )

  # Store each column of probas_matrix in a list
  # i.e. each element of the list corresponds to one context (and is a vector of proba across proba param)
  # List of T vectors of length nb_batch

  probas_list <- split(probas_matrix, row(probas_matrix))

  # Return df with probabilities for each row
  return(probas_list)
}



# GET PROBA EPSILON GREEDY -------------------------------------------------------------------------

get_proba_c_eps_greedy <- function(eps = 0.1, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch of matrices (T, K): for each policy, expected reward across arms given all contexts
  # a policy is represented by a (d, K): K vectors theta = A^-1 b of shape (d x 1)
  # we then multiply by contexts to get a (T, d) x (d, K) = (T, K)
  expected_rewards <- lapply(seq_len(nb_batch), function(t) {
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat
  }) # (T, K)

  # Convert expected_rewards (list of nb_batch matrices) into a 3D array (T × K × nb_batch)
  # T x K x nb_batch = context x arm x policy
  expected_rewards_array <- simplify2array(expected_rewards)

  # Swap last dimension (nb_batch) with second dimension (K) -> (T × nb_batch × K)
  expected_rewards_array <- aperm(expected_rewards_array, c(1, 3, 2))

  # Find max expected rewards across K for each (T, nb_batch) combo
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × nb_batch)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_batch, K))

  # For each (T, nb_batch) combo, says if arm had max expected reward or not (1 or 0)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × nb_batch × K)

  # For each (T, nb_batch) combo, count the number of best arms
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × nb_batch)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  # i.e. whether the arm chosen in the history had max expected reward or not
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × nb_batch)

  # Compute final probabilities (T × nb_batch)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

  return(proba_results)  # Returns (T × nb_batch) matrix of probabilities, one context per row
}

get_proba_c_eps_greedy_penultimate <- function(eps = 0.1, A_list, b_list, context_matrix) {
  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # Theta hat matrix of shape (d, K)
  theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[k]], b_list[[k]]), simplify = "matrix")

  # Expected rewards matrix of shape (B, K)
  expected_rewards <-  context_matrix  %*% theta_hat

  # Find max expected rewards for each row in every B
  max_rewards <- apply(expected_rewards, 1, max)  # Shape: (B)

  max_rewards_expanded <- array(max_rewards, dim = c(B, K))

  # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards == max_rewards_expanded  # Shape: (B × K)

  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, 1, sum)  # Shape: (B)

  # Compute final probabilities (B × K)
  proba_results <- (1 - eps) * ties / num_best_arms + eps / K

  return(proba_results)  # Returns (B × K) matrix of probabilities, one context per row
}

# GET PROBA UCB DISJOINT WITH EPSILON ---------------------------------------------------------

get_proba_ucb_disjoint <- function(alpha=1.0, eps = 0.1, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch of matrices (T, K): for each policy, expected reward across arms across all contexts
  # a policy is represented by a (d, K): K vectors theta = A^-1 b of shape (d x 1)
  # we then multiply by contexts to get a (T, d) x (d, K) = (T, K)
  mu <- lapply(seq_len(nb_batch), function(t) {
    theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[t]][[k]], b_list[[t]][[k]]), simplify = "matrix")
    context_matrix  %*% theta_hat # (T x K)
  }) # (T, K)

  # List of length nb_batch of matrices (T, K): for each policy, standard deviation of expected reward
  # across arms across all contexts
  variance <- lapply(seq_len(nb_batch), function(t) {
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% inv(A_list[[t]][[k]]) # (T x d)
      # We have to do that NOT to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance_terms <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
      sqrt(variance_terms)
    }, simplify = "matrix") # (T x K)
  })

  # Convert mu and variance (list of nb_batch matrices) into 3D arrays (T × K × nb_batch)
  # T x K x nb_batch = context x arm x policy
  mu_array <- simplify2array(mu)
  variance_array <- simplify2array(variance)

  # Swap last dimension (nb_batch) with second dimension (K) -> (T × nb_batch × K)
  # T x nb_batch x K = context x policy x arm
  mu_array <- aperm(mu_array, c(1, 3, 2))
  variance_array <- aperm(variance_array, c(1, 3, 2))

  expected_rewards_array <- mu_array + alpha * variance_array

  # Find max expected rewards across K for each (T, nb_batch) combo
  max_rewards <- apply(expected_rewards_array, c(1, 2), max)  # Shape: (T × nb_batch)

  max_rewards_expanded <- array(max_rewards, dim = c(nb_timesteps, nb_batch, K))

  # For each (T, nb_batch) combo, says if arm had max expected reward or not (1 or 0)
  ties <- expected_rewards_array == max_rewards_expanded  # Shape: (T × nb_batch × K)

  # For each (T, nb_batch) combo, count the number of best arms
  num_best_arms <- apply(ties, c(1, 2), sum)  # Shape: (T × nb_batch)

  # Extract chosen arm's max reward status using extract_2d_from_3d()
  # i.e. whether the arm chosen in the history had max expected reward or not
  chosen_best <- extract_2d_from_3d(ties, ind_arm)  # Shape: (T × nb_batch)

  # Compute final probabilities (T × nb_batch)
  proba_results <- (1 - eps) * chosen_best / num_best_arms + eps / K

  return(proba_results)
}

get_proba_ucb_disjoint_penultimate <- function(alpha=1.0, eps = 0.1, A_list, b_list, context_matrix) {

  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # Theta hat matrix of shape (d, K)
  theta_hat <- sapply(seq_len(K), function(k) solve(A_list[[k]], b_list[[k]]), simplify = "matrix")

  # Expected rewards matrix of shape (B, K)
  mu <-  context_matrix  %*% theta_hat

  variance_matrix <- sapply(seq_len(K), function (k) {
    semi_var <- context_matrix %*% inv(A_list[[k]]) # (B x d)
    # We have to do that NOT to end up with Xi * A_inv * t(Xj) for all combinations of i,j
    # We only want the combinations where i = j
    variance_terms <- rowSums(semi_var * context_matrix) # (vector of length B for each k)
    # for a given policy, for a given arm, we have T sigmas: one per context
    sqrt(variance_terms)
  }, simplify = "matrix") # (B x K)

  expected_rewards <- mu + alpha * variance_matrix

  # Find max expected rewards for each row in every B
  max_rewards <- apply(expected_rewards, 1, max)  # Shape: (B)

  max_rewards_expanded <- array(max_rewards, dim = c(B, K))

  # Identify ties (arms with max reward at each timestep)
  ties <- expected_rewards == max_rewards_expanded  # Shape: (B × K)

  # Count the number of best arms (how many ties per timestep)
  num_best_arms <- apply(ties, 1, sum)  # Shape: (B)

  # Compute final probabilities (B × K)
  proba_results <- (1 - eps) * ties / num_best_arms + eps / K

  return(proba_results)

}

# GET PROBA THOMPSON SAMPLING ---------------------------------------------------------------------

get_proba_thompson <- function(sigma = 0.01, A_list, b_list, contexts, ind_arm, batch_size) {
  # A_list and b_list contain the list (for agent, sim group) of theta$A and theta$b
  # Thus, each element of A_list and b_list, is itself a list (across arms) of
  # matrices A (resp. vectors b)

  # ind_arm is the vector of indices of the arms that were chosen at each t
  if (!is.integer(ind_arm)) {
    ind_arm <- as.integer(unlist(ind_arm))
  }

  K <- length(b_list[[1]])  # Number of arms
  nb_timesteps <- length(contexts)
  nb_batch <- nb_timesteps %/% batch_size

  # Convert contexts list to (T × d) matrix, put context vector in rows
  context_matrix <- do.call(rbind, contexts)

  # List of length nb_batch giving for each policy t, the array of probabilities under each context j
  # of selecting Aj
  result <- lapply(seq_len(nb_batch), function(t) {

    # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
    theta_hat <- sapply(seq_len(K), function(k) A_list[[t]][[k]] %*% b_list[[t]][[k]], simplify = "matrix")

    mean <- context_matrix  %*% theta_hat # (T x K)
    variance_matrix <- sapply(seq_len(K), function (k) {
      semi_var <- context_matrix %*% (sigma * A_list[[t]][[k]]) # (T x d)
      # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
      # We only want the combinations where i = j
      variance <- rowSums(semi_var * context_matrix) # (vector of length T for each k)
      # for a given policy, for a given arm, we have T sigmas: one per context
    }, simplify = "matrix") # (T x K)

    proba_results <- numeric(nb_timesteps)

    for (j in 1:nb_timesteps) {

      mean_k <- mean[j, ind_arm[j]]
      var_k  <-  variance_matrix[j, ind_arm[j]]
      # var_k <- max(var_k, 1e-6)

      competing_arms <- setdiff(1:K, ind_arm[j])

      mean_values <- mean[j,competing_arms]
      var_values <- variance_matrix[j, competing_arms]
      # var_values <- pmax(var_values, 1e-6)

      # Define the function for integration
      integrand <- function(x) {
        log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

        for (i in seq_along(mean_values)) {
          log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[i], sd = sqrt(var_values[i]), log.p = TRUE)
        }

        return(exp(log_p_xk))  # Convert back to probability space
      }

      # lower_bound <- mean_k - 3 * sqrt(var_k)
      # upper_bound <- mean_k + 3 * sqrt(var_k)
      all_means <- c(mean_k, mean_values)
      all_vars <- c(var_k, var_values)
      lower_bound <- min(all_means - 3 * sqrt(all_vars))
      upper_bound <- max(all_means + 3 * sqrt(all_vars))

      # Adaptive numerical integration
      prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 10, abs.tol = 1e-2)$value

      clip <- 1e-3

      proba_results[j] <- pmax(clip, pmin(prob, 1-clip))
    }

    return(proba_results)
  })

  # result is a list giving for each policy t, the array of probabilities under each context j
  # of selecting Aj
  result_matrix <- simplify2array(result) # a row should be a context, policies in columns

  return(result_matrix)
}

get_proba_thompson_penultimate <- function(sigma = 0.01, A_list, b_list, context_matrix) {

  # context_matrix is of shape (B, d)
  K <- length(b_list)  # Number of arms
  dims <- dim(context_matrix)
  B <- dims[1]

  # For penultimate policy, gives the array of probabilities under each context j (1:B)
  # of selecting arm k (1:K)

  # Solve for theta_hat (d × K): each column corresponds to theta_hat for an arm
  theta_hat <- sapply(seq_len(K), function(k) A_list[[k]] %*% b_list[[k]], simplify = "matrix")

  mean <- context_matrix  %*% theta_hat # (B x K)
  variance_matrix <- sapply(seq_len(K), function (k) {
    semi_var <- context_matrix %*% (sigma * A_list[[k]]) # (B x d)
    # We have to do that not to end up with Xi * A_inv * t(Xj) for all combinations of i,j
    # We only want the combinations where i = j
    variance <- rowSums(semi_var * context_matrix) # (vector of length B for each k)
    # for a given policy, for a given arm, we have T sigmas: one per context
  }, simplify = "matrix") # (B x K)


  result <- vector("list", K)

  for (k in 1:K) {

    proba_results <- numeric(B)

    for (j in 1:B) {

      mean_k <- mean[j, k]
      var_k  <-  variance_matrix[j, k]
      #var_k <- max(var_k, 1e-6)

      competing_arms <- setdiff(1:K, k)

      mean_values <- mean[j,competing_arms]
      var_values <- variance_matrix[j, competing_arms]
      #var_values <- pmax(var_values, 1e-6)

      # Define the function for integration
      integrand <- function(x) {
        log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

        for (i in seq_along(mean_values)) {
          log_p_xk <- log_p_xk + pnorm(x, mean = mean_values[i], sd = sqrt(var_values[i]), log.p = TRUE)
        }

        return(exp(log_p_xk))  # Convert back to probability space
      }

      # lower_bound <- mean_k - 3 * sqrt(var_k)
      # upper_bound <- mean_k + 3 * sqrt(var_k)
      all_means <- c(mean_k, mean_values)
      all_vars <- c(var_k, var_values)
      lower_bound <- min(all_means - 3 * sqrt(all_vars))
      upper_bound <- max(all_means + 3 * sqrt(all_vars))


      # Adaptive numerical integration
      prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 10, abs.tol = 1e-2)$value

      clip <- 1e-3

      proba_results[j] <- pmax(clip, pmin(prob, 1-clip))
    }

    result[[k]] <- proba_results
  }

  # result is a list giving for each arm k, the array of probabilities under each context j
  # of selecting arm k
  result_matrix <- do.call(cbind, result)
  # result_matrix <- simplify2array(result) # a row should be a context, arms in columns (B x K)

  return(result_matrix)
}


# COMPUTE ESTIMAND ------------------------------------------------------------------

compute_estimand <- function(sim_data, list_betas, policy, policy_name, batch_size, bandit) {

  # GET PARAMS OF PI_{T-1} (or PI_{T-batch_size} more generally) ------------------------
  last_timestep <- max(sim_data$t)

  last_row <- sim_data %>% filter(t == last_timestep - batch_size)

  theta_info <- last_row$theta[[1]]  # Extract the actual theta list (removing outer list structure)

  # Use A_inv if LinTSPolicy is in the string of policy_name
  key_A <- ifelse(grepl("LinTSPolicy", policy_name), "A_inv", "A")
  key_b <- "b"

  A_list <- theta_info[[key_A]]
  b_list <- theta_info[[key_b]]

  # GET BETA MATRIX FOR CURRENT SIM --------------------------------------------------

  # Safely extract simulation index
  sim_index <- theta_info$sim - 1

  beta_matrix <- list_betas[[sim_index]]  # Shape (features x arms)

  # GET INDEPENDENT CONTEXTS FROM OTHER SIMs ---------------------------------------
  B <- 1000
  d <- bandit$d
  context_matrix <- matrix(rnorm(B * d), nrow = B, ncol = d)

  # # # Take a random subset of 1000 records (if available)
  # num_samples <- min(1000, nrow(context_matrix))  # Ensure we don’t sample more than available
  # context_matrix <- context_matrix[sample(nrow(context_matrix), num_samples, replace = FALSE), , drop = FALSE]  # Shape (1000 × d)

  # Compute true linear rewards via matrix multiplication
  # True linear rewards (B × K) = (B × d) * (d × K)
  true_linear_rewards <- context_matrix %*% beta_matrix  # Shape (B x K)

  # Compute the probability matrix based on the policy name
  policy_probs <- switch(
    policy_name,
    "ContextualEpsilonGreedyPolicy" =,
    "BatchContextualEpsilonGreedyPolicy" = get_proba_c_eps_greedy_penultimate(policy$epsilon, A_list, b_list, context_matrix),  # Should be (B x K)

    "ContextualLinTSPolicy" =,
    "BatchContextualLinTSPolicy" = get_proba_thompson_penultimate(policy$sigma, A_list, b_list, context_matrix),

    "LinUCBDisjointPolicyEpsilon" =,
    "BatchLinUCBDisjointPolicyEpsilon" = get_proba_ucb_disjoint_penultimate(policy$alpha, policy$epsilon, A_list, b_list, context_matrix),

    stop("Unsupported policy_name: Choose among ContextualEpsilonGreedyPolicy, BatchContextualEpsilonGreedyPolicy,
         ContextualLinTSPolicy, BatchContextualLinTSPolicy, LinUCBDisjointPolicyEpsilon, BatchLinUCBDisjointPolicyEpsilon")
  )

  # Compute final estimand
  # B <- dim(expected_rewards)[1]
  B <- nrow(true_linear_rewards)  # Now using subset size (1000)

  estimand <- (1 / B) * sum(policy_probs * true_linear_rewards)

  return(estimand)
}


# BETAS PARAMS OF REWARD MODEL ---------------------------------------------------------------------

get_betas <- function(simulations, d, k) {
  # d: number of features
  # k: number of arms

  list_betas <- lapply(1:(simulations+1), function(i) {
    betas_matrix <- matrix(runif(d * k, -1, 1), d, k)
    betas_matrix <- betas_matrix / norm(betas_matrix, type = "2")
    return(betas_matrix)
  })

  return(list_betas)

}


# CUSTOM CONTEXTUAL LINEAR BANDIT -------------------------------------------------------------------
# store the parameters betas of the observed reward generation model

ContextualLinearBandit <- R6::R6Class(
  "ContextualLinearBandit",
  inherit = Bandit,
  class = FALSE,
  public = list(
    rewards = NULL,
    betas   = NULL,
    sigma   = NULL,
    binary  = NULL,
    weights = NULL,
    list_betas  = NULL,
    sim_id      = NULL,
    class_name = "ContextualLinearBandit",
    initialize  = function(k, d, list_betas, sigma = 0.1, binary_rewards = FALSE) {
      self$k                                    <- k
      self$d                                    <- d
      self$sigma                                <- sigma
      self$binary                               <- binary_rewards
      self$list_betas <- list_betas
    },
    post_initialization = function() {
      # self$betas                                <- matrix(runif(self$d*self$k, -1, 1), self$d, self$k)
      # self$betas                                <- self$betas / norm(self$betas, type = "2")
      # list_betas                                <<- c(list_betas, list(self$betas))
      self$betas <- self$list_betas[[self$sim_id]]

    },
    get_context = function(t) {

      X                                         <- rnorm(self$d)
      self$weights                              <- X %*% self$betas
      reward_vector                             <- self$weights + rnorm(self$k, sd = self$sigma)

      if (isTRUE(self$binary)) {
        self$rewards                            <- rep(0,self$k)
        self$rewards[which_max_tied(reward_vector)] <- 1
      } else {
        self$rewards                            <- reward_vector
      }
      context <- list(
        k = self$k,
        d = self$d,
        X = X
      )
    },
    get_reward = function(t, context_common, action) {
      rewards        <- self$rewards
      optimal_arm    <- which_max_tied(self$weights)
      reward         <- list(
        reward                   = rewards[action$choice],
        optimal_arm              = optimal_arm,
        optimal_reward           = rewards[optimal_arm]
      )
    }
  )
)

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

# BATCH VERSION OF CONTEXTUAL LINEAR POLICIES ----------------------------------------------------

BatchContextualEpsilonGreedyPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    epsilon = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchContextualEpsilonGreedyPolicy",
    initialize = function(epsilon = 0.1, batch_size=1) {
      super$initialize()
      self$epsilon <- epsilon
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      d <- context_params$d
      k <- context_params$k
      self$theta_to_arms <- list('A' = diag(1,d,d), 'b' = rep(0,d))
      self$A_cc <- replicate(k, diag(1, d, d), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,d), simplify = FALSE)
    },
    get_action = function(t, context) {

      if (runif(1) > self$epsilon) {
        expected_rewards <- rep(0.0, context$k)
        for (arm in 1:context$k) {
          Xa         <- get_arm_context(context, arm)
          A          <- self$theta$A[[arm]]
          b          <- self$theta$b[[arm]]
          A_inv      <- inv(A)
          theta_hat  <- A_inv %*% b
          expected_rewards[arm] <- Xa %*% theta_hat
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
      Xa     <- get_arm_context(context, arm)

      self$A_cc[[arm]] <- self$A_cc[[arm]] + outer(Xa, Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)


BatchLinUCBDisjointPolicyEpsilon <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    alpha = NULL,
    epsilon = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchLinUCBDisjointPolicyEpsilon",
    initialize = function(alpha = 1.0, epsilon=0.1, batch_size = 1) {
      super$initialize()
      self$alpha <- alpha
      self$epsilon <- epsilon
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      ul <- length(context_params$unique)
      k <- context_params$k
      self$theta_to_arms <- list('A' = diag(1,ul,ul), 'b' = rep(0,ul))
      self$A_cc <- replicate(k, diag(1, ul, ul), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,ul), simplify = FALSE)
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

      self$A_cc[[arm]] <- self$A_cc[[arm]] + outer(Xa, Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)

BatchContextualLinTSPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    sigma = NULL,
    batch_size = NULL,
    A_cc = NULL,
    b_cc = NULL,
    class_name = "BatchContextualLinTSPolicy",
    initialize = function(v = 0.2, batch_size=1) {
      super$initialize()
      self$sigma   <- v^2
      self$batch_size <- batch_size
      self$A_cc <- A_cc
      self$b_cc <- b_cc
    },
    set_parameters = function(context_params) {
      ul                 <- length(context_params$unique)
      k <- context_params$k
      self$theta_to_arms <- list('A_inv' = diag(1, ul, ul), 'b' = rep(0, ul))
      self$A_cc <- replicate(k, diag(1, ul, ul), simplify = FALSE)
      self$b_cc <- replicate(k, rep(0,ul), simplify = FALSE)
    },
    get_action = function(t, context) {
      expected_rewards           <- rep(0.0, context$k)
      for (arm in 1:context$k) {
        Xa                       <- get_arm_context(context, arm, context$unique)
        A_inv                    <- self$theta$A_inv[[arm]]
        b                        <- self$theta$b[[arm]]
        theta_hat                <- A_inv %*% b
        sigma_hat                <- self$sigma * A_inv
        theta_tilde              <- as.vector(contextual::mvrnorm(1, theta_hat, sigma_hat))
        expected_rewards[arm]    <- Xa %*% theta_tilde
      }
      action$choice              <- which_max_tied(expected_rewards)
      action
    },
    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa    <- get_arm_context(context, arm, context$unique)

      self$A_cc[[arm]] <- sherman_morrisson(self$A_cc[[arm]],Xa)
      self$b_cc[[arm]] <- self$b_cc[[arm]] + reward * Xa

      if (t %% self$batch_size == 0) {
        self$theta$A_inv <- self$A_cc
        self$theta$b <- self$b_cc
      }

      self$theta
    }
  )
)

# SIMULATOR with reward beta parameters retrieval ----------------------------------------------------

Simulator <- R6::R6Class(
  "Simulator",
  class = FALSE,
  public = list(
    agents = NULL,
    workers = NULL,
    agent_count = NULL,
    horizon = NULL,
    simulations = NULL,
    worker_max = NULL,
    internal_history = NULL,
    save_context = NULL,
    save_theta = NULL,
    do_parallel = NULL,
    sims_and_agents_list = NULL,
    t_over_sims = NULL,
    set_seed = NULL,
    progress_file = NULL,
    log_interval = NULL,
    save_interval = NULL,
    include_packages = NULL,
    outfile = NULL,
    global_seed = NULL,
    chunk_multiplier = NULL,
    policy_time_loop = NULL,
    cl = NULL,
    initialize = function(agents,
                          horizon = 100L,
                          simulations = 100L,
                          save_context = FALSE,
                          save_theta = FALSE,
                          do_parallel = TRUE,
                          worker_max = NULL,
                          set_seed = 0,
                          save_interval = 1,
                          progress_file = FALSE,
                          log_interval = 1000,
                          include_packages = NULL,
                          t_over_sims = FALSE,
                          chunk_multiplier = 1,
                          policy_time_loop = FALSE) {

      # save current seed
      self$global_seed <- contextual::get_global_seed()

      if (!is.list(agents)) agents <- list(agents)

      self$progress_file <- progress_file
      self$log_interval <- as.integer(log_interval)
      self$horizon <- as.integer(horizon)
      self$simulations <- as.integer(simulations)
      self$save_theta <- save_theta
      self$save_context <- save_context
      self$agents <- agents
      self$agent_count <- length(agents)
      self$worker_max <- worker_max
      self$do_parallel <- do_parallel
      self$t_over_sims <- t_over_sims
      self$set_seed <- set_seed
      self$save_interval <- as.integer(save_interval)
      self$include_packages <- include_packages
      self$chunk_multiplier <- as.integer(chunk_multiplier)
      self$policy_time_loop <- policy_time_loop

      self$reset()
    },
    reset = function() {
      set.seed(self$set_seed)
      self$workers <- 1

      # create or clear log files
      if (self$progress_file) {
        cat(paste0(""), file = "workers_progress.log", append = FALSE)
        cat(paste0(""), file = "agents_progress.log", append = FALSE)
        cat(paste0(""), file = "parallel.log", append = FALSE)
        self$outfile <- "parallel.log"
      }

      # (re)create history data and meta data tables
      self$internal_history <- History$new()
      self$internal_history$set_meta_data("horizon",self$horizon)
      self$internal_history$set_meta_data("agents",self$agent_count)
      self$internal_history$set_meta_data("simulations",self$simulations)
      self$internal_history$set_meta_data("sim_start_time",format(Sys.time(), "%a %b %d %X %Y"))

      # unique policy name creation
      agent_name_list <- list()
      for (agent_index in 1L:self$agent_count) {
        current_agent_name <- self$agents[[agent_index]]$name
        agent_name_list <- c(agent_name_list,current_agent_name)
        current_agent_name_occurrences <-
          length(agent_name_list[agent_name_list == current_agent_name])
        if (current_agent_name_occurrences > 1) {
          self$agents[[agent_index]]$name <-
            paste0(current_agent_name,'.',current_agent_name_occurrences)
        }
        agent_name <-  self$agents[[agent_index]]$name
        bandit_name <- self$agents[[agent_index]]$bandit$class_name
        policy_name <- self$agents[[agent_index]]$policy$class_name
        self$internal_history$set_meta_data("bandit", bandit_name , group = "sim", agent_name = agent_name)
        self$internal_history$set_meta_data("policy", policy_name , group = "sim", agent_name = agent_name)
      }
    },
    run = function() {
      # set parallel or serial processing
      `%fun%` <- foreach::`%do%`

      # nocov start
      if (self$do_parallel) {
        self$register_parallel_backend()
        `%fun%` <- foreach::`%dopar%`

        # If Microsoft R, set MKL threads to 1

        # Due to an unresolved incompatibility between MRAN and RStudio:
        # https://github.com/rstudio/rstudio/issues/5933
        # https://social.technet.microsoft.com/Forums/en-US/2791e896-c284-4330-88f2-2dcd4acea074
        # setting MKL threads to 1 is disabled when running from RStudio.

        isRStudio <- Sys.getenv("RSTUDIO") == "1"
        if (!isRStudio && "RevoUtilsMath" %in% rownames(installed.packages())) {
          RevoUtilsMath::setMKLthreads(1)
        }
      }
      # nocov end

      # create a list of all sims (sims*agents), to be divided into chunks
      index <- 1
      sims_and_agents_list <- vector("list", self$simulations*self$agent_count)
      for (sim_index in 1L:self$simulations) {
        for (agent_index in 1L:self$agent_count) {
          sims_and_agents_list[[index]] <-
            list(agent_index = agent_index, sim_index   = sim_index)
          index <- index + 1
        }
      }

      # copy variables used in parallel processing to local environment
      horizon                  <- self$horizon
      agent_count              <- self$agent_count
      save_context             <- self$save_context
      save_theta               <- self$save_theta
      progress_file            <- self$progress_file
      save_interval            <- self$save_interval
      log_interval             <- self$log_interval
      t_over_sims              <- self$t_over_sims
      set_seed                 <- self$set_seed
      agents                   <- self$agents
      include_packages         <- self$include_packages
      policy_time_loop          <- self$policy_time_loop

      # calculate chunk size
      if (length(sims_and_agents_list) <= self$workers) {
        chunk_divider <- length(sims_and_agents_list)
      } else {
        chunk_divider <- self$workers * self$chunk_multiplier
      }
      # split sims vector into chuncks
      sa_iterator <- itertools::isplitVector(sims_and_agents_list, chunks = chunk_divider)
      # include packages that are used in parallel processes
      par_packages <- c(c("data.table","iterators","itertools"),include_packages)

      # some info messages
      message(paste("Simulation horizon:",horizon))
      message(paste("Number of simulations:",length(sims_and_agents_list)))
      message(paste("Number of batches:",chunk_divider))
      message("Starting main loop.")

      # start running the main simulation loop
      private$start_time <- Sys.time()
      foreach_results <- foreach::foreach(
        sims_agent_list = sa_iterator,
        i = iterators::icount(),
        .inorder = TRUE,
        .export = c("History","Formula"),
        .noexport = c("sims_and_agents_list","internal_history","sa_iterator"),
        .packages = par_packages
      ) %fun% {
        index <- 1L
        sim_agent_counter <- 0
        sim_agent_total <- length(sims_agent_list)

        # TODO: Can be done smarter and cleaner?
        multiplier <- 1
        for (sim_agent_index in sims_agent_list) {
          sim_agent <- agents[[sim_agent_index$agent_index]]
          if(isTRUE(sim_agent$bandit$arm_multiply))
            if(multiplier < sim_agent$bandit$k)
              multiplier <- sim_agent$bandit$k
        }
        allocate_space <- floor((horizon * sim_agent_total * multiplier) / save_interval) + sim_agent_total

        local_history <- History$new( allocate_space,
                                      save_context,
                                      save_theta)

        for (sim_agent_index in sims_agent_list) {
          sim_agent <- agents[[sim_agent_index$agent_index]]$clone(deep = TRUE)

          sim_agent$sim_index <- sim_agent_index$sim_index
          sim_agent$agent_index <- sim_agent_index$agent_index

          ###############################################################################################

          # ✅ Set sim_id explicitly for the bandit
          sim_agent$bandit$sim_id <- sim_agent_index$sim_index

          ###############################################################################################


          sim_agent_counter <- sim_agent_counter + 1
          if (isTRUE(progress_file)) {
            sim_agent$progress_file <- TRUE
            sim_agent$log_interval <- log_interval
            cat(paste0("[",format(Sys.time(), format = "%H:%M:%OS6"),"] ",
                       "        0 > init - ",sprintf("%-20s", sim_agent$name),
                       " worker ", i,
                       " at sim ", sim_agent_counter,
                       " of ", sim_agent_total,"\n"),
                file = "workers_progress.log", append = TRUE)
          }
          simulation_index <- sim_agent$sim_index
          agent_name <- sim_agent$name
          local_curent_seed <- simulation_index + set_seed * 42
          set.seed(local_curent_seed)
          sim_agent$bandit$post_initialization()
          sim_agent$policy$post_initialization()
          if(isTRUE(sim_agent$bandit$arm_multiply)) {
            if(policy_time_loop)
              horizon_loop <- horizon
            else
              horizon_loop <- horizon * sim_agent$bandit$k
            data_length <- horizon * sim_agent$bandit$k
          } else {
            horizon_loop <- horizon
            data_length <- horizon
          }
          set.seed(local_curent_seed + 1e+06)
          sim_agent$bandit$generate_bandit_data(n = data_length)

          if (isTRUE(t_over_sims)) sim_agent$set_t(as.integer((simulation_index - 1L) * horizon_loop))
          step <- list()

          loop_time <- 0L
          while (loop_time < horizon_loop) {
            step <- sim_agent$do_step()
            if(isTRUE(policy_time_loop)) {
              loop_time <- step[[5]]
            } else {
              loop_time <- loop_time + 1L
            }
            if (!is.null(step[[3]]) && ((step[[5]] == 1) || (step[[5]] %% save_interval == 0))) {
              local_history$insert(
                index,                                         #index
                step[[5]],                                     #policy_t
                step[[1]][["k"]],                              #k
                step[[1]][["d"]],                              #d
                step[[2]],                                     #action
                step[[3]],                                     #reward
                agent_name,                                    #agentname
                simulation_index,                              #sim
                if (save_context) step[[1]][["X"]] else NA,    #context
                if (save_theta) step[[4]] else NA              #theta
              )
              index <- index + 1L
            }
          }
        }
        sim_agent$bandit$final()
        local_history$data[t!=0]
      }

      # bind all results
      foreach_results <- data.table::rbindlist(foreach_results)
      foreach_results[, agent := factor(agent)]
      self$internal_history$set_data_table(foreach_results[sim > 0 & t > 0], auto_stats = FALSE)
      rm(foreach_results)
      private$end_time <- Sys.time()
      gc()
      message("Finished main loop.")

      self$internal_history$set_meta_data("sim_end_time",format(Sys.time(), "%a %b %d %X %Y"))
      formatted_duration <- contextual::formatted_difftime(private$end_time - private$start_time)
      self$internal_history$set_meta_data("sim_total_duration", formatted_duration)
      message(paste0("Completed simulation in ",formatted_duration))

      start_time_stats <- Sys.time()
      message("Computing statistics.")
      # update statistics TODO: not always necessary, add option arg to class?
      self$internal_history$update_statistics()

      # load global seed
      .Random.seed <- self$global_seed

      # set meta data and messages
      self$stop_parallel_backend()
      self$internal_history
    },
    register_parallel_backend = function() {
      # nocov start
      # setup parallel backend
      message("Setting up parallel backend.")
      nr_cores <- parallel::detectCores()
      if (nr_cores >= 3) self$workers <- nr_cores - 1
      if (!is.null(self$worker_max)) {
        if (self$workers > self$worker_max) self$workers <- self$worker_max
      }

      # make sure no leftover processes
      doParallel::stopImplicitCluster()


      if(!is.null(self$outfile)) {
        self$cl <- parallel::makeCluster(self$workers, useXDR = FALSE, type = "PSOCK",
                                         methods = FALSE, setup_timeout = 30, outfile = self$outfile)
      } else {
        self$cl <- parallel::makeCluster(self$workers, useXDR = FALSE, type = "PSOCK",
                                         methods = FALSE, setup_timeout = 30)
      }

      message(paste0("Cores available: ",nr_cores))
      message(paste0("Workers assigned: ",self$workers))
      doParallel::registerDoParallel(self$cl)
      # nocov end
    },
    stop_parallel_backend = function() {
      # nocov start
      if (self$do_parallel) {
        try({
          parallel::stopCluster(self$cl)
        })
        doParallel::stopImplicitCluster()
      }
      # nocov end
    },
    finalize = function() {
      # set global seed back to value before
      contextual::set_global_seed(self$global_seed)
      #closeAllConnections()
    }
  ),
  private = list(
    start_time = NULL,
    end_time = NULL
  ),
  active = list(
    history = function(value) {
      if (missing(value)) {
        self$internal_history
      } else {
        warning("## history$data is read only", call. = FALSE)
      }
    }
  )
)
