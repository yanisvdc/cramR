# Load necessary libraries
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R
library(doParallel)
library(foreach)

# Declare global variables to suppress devtools::check warnings
utils::globalVariables(c("X_cumul", "D_cumul", "Y_cumul", "data_cumul", "."))

#' On-policy CRAM Bandit
#'
#' This function performs simulation for cram bandit policy learning and evaluation.
#'
#' @param pi for each row j, column t, depth a, gives pi_t(Xj, a)
#' @param arm arms selected at each time step
#' @param reward rewards at each time step
#'
#' @return A **list** containing:
#'   \item{final_ml_model}{The final trained ML model.}
#'   \item{losses}{A matrix of losses where each column represents a batch's trained model. The first column contains zeros (baseline model).}
#'   \item{batch_indices}{The indices of observations in each batch.}
#'
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{keras_model_sequential}}
#' @import contextual
#' @importFrom grf causal_forest
#' @importFrom glmnet cv.glmnet
#' @importFrom keras keras_model_sequential layer_dense compile fit
#' @importFrom stats glm predict qnorm rbinom rnorm
#' @importFrom magrittr %>%
#' @import data.table
#' @importFrom parallel makeCluster detectCores stopCluster clusterExport
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach %dopar% foreach
#' @importFrom stats var
#' @importFrom grDevices col2rgb
#' @importFrom stats D

cram_bandit_sim <- function(data, formula=NULL, batch,
                        parallelize_batch = FALSE, loss_name = NULL,
                        caret_params = NULL, custom_fit = NULL,
                        custom_predict = NULL, custom_loss = NULL,
                        n_cores = detectCores() - 1) {




}





# install.packages("remotes")
# remotes::install_github("Nth-iteration-labs/contextual")

# Load the necessary library
library(contextual)

# Simulation settings
horizon <- 1000  # Number of rounds
simulations <- 1 # Number of simulations
k <- 5           # Number of arms
d <- 3           # Number of contextual features

# Define a Linear Contextual Bandit
bandit <- ContextualLinearBandit$new(k = k, d = d)

# Define Linear Thompson Sampling Policy
policy <- ContextualLinTSPolicy$new(v = 0.1)  # v controls exploration-exploitation

# Create an agent with the policy and bandit
agent <- Agent$new(policy = policy, bandit = bandit)

# Run the simulation
simulator <- Simulator$new(agents = agent, horizon = horizon, simulations = simulations)
simulator$run()

# Retrieve and summarize history
history <- simulator$history
summary(history)

# Plot cumulative rewards over time
plot(history)


library(contextual)

horizon       <- 100L
simulations   <- 100L

bandit        <- ContextualLinearBandit$new(k = 4, d = 3, sigma = 0.3)

# Linear CMAB policies comparison

agents <- list(Agent$new(ContextualLinTSPolicy$new(0.1), bandit, "LinTS"))

simulation     <- Simulator$new(agents, horizon, simulations, do_parallel = TRUE,
                                save_context=TRUE, save_theta = TRUE)

history        <- simulation$run()

plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")



res <- history$data


param_lookup <- res$theta




## FINAL HERE ---------------------------------------------------------

library(stats)
library(pracma)  # For numerical integration
library(MASS)    # For multivariate normal sampling (mvrnorm)

CramContextualEpsilonGreedyPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    epsilon = NULL,
    class_name = "CramContextualEpsilonGreedyPolicy",

    initialize = function(epsilon = 0.1) {
      super$initialize()
      self$epsilon <- epsilon
    },

    set_parameters = function(context_params) {
      d <- context_params$d
      self$theta_to_arms <- list('A' = diag(1, d, d), 'b' = rep(0, d))
    },

    get_action = function(t, context) {
      k <- context$k  # Number of arms

      if (runif(1) > self$epsilon) {
        # EXPLOIT: Select the arm with the highest estimated reward
        expected_rewards <- rep(0.0, k)

        for (arm in 1:k) {
          Xa        <- get_arm_context(context, arm)  # Get feature vector for arm
          A         <- self$theta$A[[arm]]
          b         <- self$theta$b[[arm]]
          A_inv     <- inv(A)  # Compute inverse of A
          theta_hat <- A_inv %*% b  # Compute estimated theta

          expected_rewards[arm] <- Xa %*% theta_hat
        }

        ties <- unlist(expected_rewards, FALSE, FALSE)
        ties <- seq_along(ties)[ties == max(ties)]
        num_best_arms <- length(ties)

        self$action$choice  <- which_max_tied(expected_rewards)

        # Corrected propensity calculation
        self$action$propensity <- (1 - self$epsilon) / num_best_arms + self$epsilon*(1/context$k)

      } else {
        # EXPLORE: Choose a random arm
        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)

        # Exploration probability
        self$action$propensity <- self$epsilon*(1/context$k)
      }

      self$action
    },

    set_reward = function(t, context, action, reward) {
      arm    <- action$choice
      reward <- reward$reward
      Xa     <- get_arm_context(context, arm)

      # Update A and b for the chosen arm
      inc(self$theta$A[[arm]]) <- outer(Xa, Xa)  # Update A
      inc(self$theta$b[[arm]]) <- reward * Xa    # Update b

      return(self$theta)
    }
  )
)



CramLinUCBDisjointPolicy <- R6::R6Class(
  portable = FALSE,
  class = FALSE,
  inherit = Policy,
  public = list(
    epsilon = NULL,
    alpha = NULL,
    class_name = "CramLinUCBDisjointPolicy",
    initialize = function(alpha = 1.0, epsilon = 0.0) {
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

        ties <- unlist(expected_rewards, FALSE, FALSE)
        ties <- seq_along(ties)[ties == max(ties)]
        num_best_arms <- length(ties)

        self$action$choice  <- which_max_tied(expected_rewards)

        # Corrected propensity calculation
        self$action$propensity <- (1 - self$epsilon) / num_best_arms + self$epsilon*(1/context$k)

      } else {
        # EXPLORE: Choose a random arm
        self$action$choice        <- sample.int(context$k, 1, replace = TRUE)

        # Exploration probability
        self$action$propensity <- self$epsilon*(1/context$k)
      }
      self$action
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





library(contextual)

horizon       <- 100L
simulations   <- 100L

bandit        <- ContextualLinearBandit$new(k = 4, d = 3, sigma = 0.3)

# Linear CMAB policies comparison

agents <- list(Agent$new(EpsilonGreedyPolicy$new(0.1), bandit, "EGreedy"),
               Agent$new(CramContextualEpsilonGreedyPolicy$new(0.1), bandit, "cramcEGreedy"),
               Agent$new(ContextualEpsilonGreedyPolicy$new(0.1), bandit, "cEGreedy"),
               Agent$new(ContextualLinTSPolicy$new(0.1), bandit, "LinTS"),
               Agent$new(CramContextualLinTSPolicy$new(0.1), bandit, "cramLinTS"),
               Agent$new(CramLinUCBDisjointPolicy$new(0.6, 0.5), bandit, "CramLinUCB"),
               Agent$new(LinUCBDisjointPolicy$new(0.6), bandit, "LinUCB"))

simulation     <- Simulator$new(agents, horizon, simulations,
                                do_parallel = TRUE, save_theta = TRUE,
                                save_context = TRUE)

history        <- simulation$run()

plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")


res <- history$data

param_lookup <- history$data$theta



res$t






# install.packages("pracma")
library(stats)
library(pracma)  # For numerical integration

p_max_single_arm <- function(theta, sigma, k) {
  K <- length(theta)  # Number of arms

  # Define the function for integration
  integrand <- function(x) {
    p_xk <- dnorm(x, mean = theta[k], sd = sigma[k])  # P(X_k = x)
    cdf_prod <- prod(pnorm(x, mean = theta[-k], sd = sigma[-k]))  # P(X_i < x) for all i ≠ k
    return(p_xk * cdf_prod)
  }

  # Adaptive numerical integration
  prob <- integral(integrand, -Inf, Inf) # Adaptive quadrature integration

  return(prob)
}

# Example Usage
theta_example <- c(1.0, 0.5, -0.2)  # Example means for 3 arms
sigma_example <- c(0.3, 0.4, 0.2)   # Example standard deviations for 3 arms
chosen_arm <- 1  # Compute probability for arm 1 (1-based index in R)

p_max_single_arm(theta_example, sigma_example, chosen_arm)















library(stats)
library(pracma)  # For numerical integration
library(MASS)    # For multivariate normal sampling (mvrnorm)

p_max_single_arm <- function(theta_hat, sigma_hat, Xa, k) {
  K <- length(theta_hat)  # Number of arms

  # Define the function for integration
  integrand <- function(x) {
    # P( X_k = x ) where X_k ~ N(Xa_k^T theta_hat_k, Xa_k^T sigma_hat_k Xa_k)
    mean_k <- Xa[k, ] %*% theta_hat[[k]]  # Xa * theta_hat for arm k
    var_k  <- Xa[k, ] %*% sigma_hat[[k]] %*% Xa[k, ]  # Xa * sigma_hat * Xa^T

    p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k))  # P(X_k = x)

    # Compute the probability that all other arms have a smaller value
    cdf_prod <- prod(sapply(setdiff(1:K, k), function(i) {
      mean_i <- Xa[i, ] %*% theta_hat[[i]]
      var_i  <- Xa[i, ] %*% sigma_hat[[i]] %*% Xa[i, ]
      pnorm(x, mean = mean_i, sd = sqrt(var_i))  # P(X_i < x)
    }))

    return(p_xk * cdf_prod)
  }

  # Adaptive numerical integration
  prob <- integral(integrand, -Inf, Inf)  # Efficient computation

  return(prob)
}

# Example Usage
theta_hat_example <- list(
  c(1.0, -0.5),  # Theta for arm 1
  c(0.5, 0.2),   # Theta for arm 2
  c(-0.2, 0.8)   # Theta for arm 3
)

sigma_hat_example <- list(
  matrix(c(0.3, 0.1, 0.1, 0.2), 2, 2),  # Covariance for arm 1
  matrix(c(0.4, 0.05, 0.05, 0.3), 2, 2), # Covariance for arm 2
  matrix(c(0.2, 0.02, 0.02, 0.25), 2, 2) # Covariance for arm 3
)

Xa_example <- matrix(c(1, 2,  # Context vector for arm 1
                       0.5, 1,  # Context vector for arm 2
                       -1, 0.5), 3, 2, byrow = TRUE)

chosen_arm <- 1  # Compute probability for arm 1

p_max_single_arm(theta_hat_example, sigma_hat_example, Xa_example, chosen_arm)







library(stats)
library(pracma)  # For numerical integration
library(MASS)    # For multivariate normal sampling (mvrnorm)

p_max_single_arm <- function(mu, Sigma, Xa, k) {
  K <- length(mu)  # Number of arms

  # Define the function for integration
  integrand <- function(x) {
    # Compute the transformed mean and variance for the chosen arm
    mean_k <- Xa[k, ] %*% mu[[k]]
    var_k  <- Xa[k, ] %*% Sigma[[k]] %*% Xa[k, ]

    log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF

    # Compute log-probabilities that all other arms have a lower reward
    log_cdf_sum <- sum(sapply(setdiff(1:K, k), function(i) {
      mean_i <- Xa[i, ] %*% mu[[i]]
      var_i  <- Xa[i, ] %*% Sigma[[i]] %*% Xa[i, ]
      pnorm(x, mean = mean_i, sd = sqrt(var_i), log.p = TRUE)  # Log-CDF
    }))

    return(exp(log_p_xk + log_cdf_sum))  # Convert back to probability space
  }

  # Adaptive numerical integration
  prob <- integral(integrand, -Inf, Inf)

  return(prob)
}

# Example Usage
theta_hat_example <- list(
  c(1.0, -0.5),  # Theta for arm 1
  c(0.5, 0.2),   # Theta for arm 2
  c(-0.2, 0.8)   # Theta for arm 3
)

sigma_hat_example <- list(
  matrix(c(0.3, 0.1, 0.1, 0.2), 2, 2),  # Covariance for arm 1
  matrix(c(0.4, 0.05, 0.05, 0.3), 2, 2), # Covariance for arm 2
  matrix(c(0.2, 0.02, 0.02, 0.25), 2, 2) # Covariance for arm 3
)

Xa_example <- matrix(c(1, 2,  # Context vector for arm 1
                       0.5, 1,  # Context vector for arm 2
                       -1, 0.5), 3, 2, byrow = TRUE)

chosen_arm <- 1

p_max_single_arm(theta_hat_example, sigma_hat_example, Xa_example, chosen_arm)





library(contextual)
library(data.table)

horizon       <- 100L
simulations   <- 100L

bandit        <- ContextualLinearBandit$new(k = 4, d = 3, sigma = 0.3)

# Linear CMAB policies comparison

# agents <- list(Agent$new(ContextualEpsilonGreedyPolicy$new(0.1), bandit, "cEGreedy"),
#                Agent$new(ContextualLinTSPolicy$new(0.1), bandit, "LinTS"),
#                Agent$new(LinUCBDisjointPolicy$new(0.6), bandit, "LinUCB"))

# agents <- list(Agent$new(ContextualEpsilonGreedyPolicy$new(0.1), bandit, "cEGreedy"))

agents <- list(Agent$new(ContextualLinTSPolicy$new(0.1), bandit, "LinTS"))

simulation     <- Simulator$new(agents, horizon, simulations,
                                do_parallel = TRUE, save_theta = TRUE,
                                save_context = TRUE)

history        <- simulation$run()

plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")


res <- history$data

param_lookup <- history$data$theta


# Retrieve d (feature dimension) from any row
d_value <- res$d[1]  # Assuming d is constant across rows

# Dynamically select the X.1 to X.d columns
X_columns <- paste0("X.", 1:d_value)

# Subset res, keeping only the relevant columns
res_subset <- res[, c("agent", "sim", "t", "choice", "reward", "theta", X_columns), with = FALSE]

# Convert X.1, ..., X.d into a single list-column `context`
res_subset[, context := lapply(1:.N, function(i) unlist(.SD[i, ], use.names = FALSE)), .SDcols = X_columns]

# Remove original X.1, ..., X.d columns after storing in `context`
res_subset[, (X_columns) := NULL]


# # Retrieve k (number of arms) dynamically from any row
# k_value <- res$k[1]
#
# # Convert context vectors into k × d matrices
# res_subset[, context := lapply(context, function(vec) matrix(rep(vec, each = k_value), nrow = k_value, byrow = TRUE))]


# Step 1: Convert NULL values in theta to NA (without modifying res_subset)
res_subset[, theta_na := lapply(theta, function(x) if (is.null(x)) NA else x)]

# Step 2: Count the number of NA values in theta_na per simulation
null_counts <- res_subset[, .(num_nulls = sum(is.na(theta_na))), by = sim]

# Step 3: Identify simulations to drop (where num_nulls >= 1)
sims_to_remove <- null_counts[num_nulls >= 1, sim]

# Step 4: Remove these simulations from res_subset in place
res_subset <- res_subset[!sim %in% sims_to_remove]

# Step 5: Drop the temporary column (optional)
res_subset[, theta_na := NULL]


# Ensure data is sorted correctly by agent, sim, t
setorder(res_subset, agent, sim, t)


# PROBA CALCULATION FUNCTIONS ------------------------------------------------------------

# CONTEXTUAL EPSILON GREEDY -------------------------------------------------------

get_proba_c_eps_greedy <- function(eps = 0.1, A, b, contexts, ind_arm) {
  # ind_arm is the index of the arm that was chosen

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
    if (ind_arm %in% ties) {
      proba_results[t] <- (1 - eps) / length(ties) + eps / K
    } else {
      proba_results[t] <- eps / K
    }
  }

  return(proba_results)
}

# UCB -------------------------------------------------------------------------------------



# THOMPSON SAMPLING -----------------------------------------------------------------------

# CramContextualLinTSPolicy <- R6::R6Class(
#   portable = FALSE,
#   class = FALSE,
#   inherit = Policy,
#   public = list(
#     sigma = NULL,
#     class_name = "CramContextualLinTSPolicy",
#     initialize = function(v = 0.2) {
#       super$initialize()
#       self$sigma   <- v^2
#     },
#     set_parameters = function(context_params) {
#       ul                 <- length(context_params$unique)
#       self$theta_to_arms <- list('A_inv' = diag(1, ul, ul), 'b' = rep(0, ul))
#     },
#     get_action = function(t, context) {
#       expected_rewards           <- rep(0.0, context$k)
#       means <- vector("list", context$k)  # Store theta_hat as a list of vectors
#       sigmas <- vector("list", context$k) # Store sigma_hat as a list of matrices
#       contexts <- matrix(0, nrow = context$k, ncol = length(context$unique)) # Store Xa as a matrix
#
#       for (arm in 1:context$k) {
#         Xa                       <- get_arm_context(context, arm, context$unique)
#         A_inv                    <- self$theta$A_inv[[arm]]
#         b                        <- self$theta$b[[arm]]
#
#         theta_hat                <- A_inv %*% b
#         sigma_hat                <- self$sigma * A_inv
#         theta_tilde              <- as.vector(contextual::mvrnorm(1, theta_hat, sigma_hat))
#         expected_rewards[arm]    <- Xa %*% theta_tilde
#
#         means[[arm]] <- as.vector(theta_hat)  # Store theta_hat correctly
#         sigmas[[arm]] <- sigma_hat  # Store sigma_hat as a matrix
#         contexts[arm, ] <- Xa  # Store Xa as a row in the matrix
#       }
#       self$action$choice              <- which_max_tied(expected_rewards)
#
#       ind_arm <- self$action$choice
#       self$action$propensity <- self$get_prob(means, sigmas, contexts, ind_arm)
#
#       self$action
#     },
#     set_reward = function(t, context, action, reward) {
#       arm    <- action$choice
#       reward <- reward$reward
#
#       Xa    <- get_arm_context(context, arm, context$unique)
#
#       self$theta$A_inv[[arm]]  <- sherman_morrisson(self$theta$A_inv[[arm]],Xa)
#       self$theta$b[[arm]]      <- self$theta$b[[arm]] + reward * Xa
#
#       self$theta
#     },
#     get_prob = function(mu, Sigma, Xa, k) {
#       K <- length(mu)  # Number of arms
#
#       # Define the function for integration
#       integrand <- function(x) {
#         # Compute the transformed mean and variance for the chosen arm
#         mean_k <- Xa[k, ] %*% mu[[k]]
#         var_k  <- Xa[k, ] %*% Sigma[[k]] %*% Xa[k, ]
#
#         log_p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k), log = TRUE)  # Log-PDF
#
#         # Compute log-probabilities that all other arms have a lower reward
#         log_cdf_sum <- sum(sapply(setdiff(1:K, k), function(i) {
#           mean_i <- Xa[i, ] %*% mu[[i]]
#           var_i  <- Xa[i, ] %*% Sigma[[i]] %*% Xa[i, ]
#           pnorm(x, mean = mean_i, sd = sqrt(var_i), log.p = TRUE)  # Log-CDF
#         }))
#
#         return(exp(log_p_xk + log_cdf_sum))  # Convert back to probability space
#       }
#
#       # Adaptive numerical integration
#       prob <- pracma::integral(integrand, -Inf, Inf)
#
#       return(prob)
#     }
#   )
# )


# get_proba_thompson <- function(theta_hat, sigma_hat, Xa, k) {
#   K <- length(theta_hat)  # Number of arms
#
#   # Define the function for integration
#   integrand <- function(x) {
#     # P( X_k = x ) where X_k ~ N(Xa_k^T theta_hat_k, Xa_k^T sigma_hat_k Xa_k)
#     mean_k <- Xa[k, ] %*% theta_hat[[k]]  # Xa * theta_hat for arm k
#     var_k  <- Xa[k, ] %*% sigma_hat[[k]] %*% Xa[k, ]  # Xa * sigma_hat * Xa^T
#
#     p_xk <- dnorm(x, mean = mean_k, sd = sqrt(var_k))  # P(X_k = x)
#
#     # Compute the probability that all other arms have a smaller value
#     cdf_prod <- prod(sapply(setdiff(1:K, k), function(i) {
#       mean_i <- Xa[i, ] %*% theta_hat[[i]]
#       var_i  <- Xa[i, ] %*% sigma_hat[[i]] %*% Xa[i, ]
#       pnorm(x, mean = mean_i, sd = sqrt(var_i))  # P(X_i < x)
#     }))
#
#     return(p_xk * cdf_prod)
#   }
#
#   # Adaptive numerical integration
#   prob <- integral(integrand, -Inf, Inf)  # Efficient computation
#
#   return(prob)
# }

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

    lower_bound <- mean_k - 10 * sqrt(var_k)
    upper_bound <- mean_k + 10 * sqrt(var_k)

    # Adaptive numerical integration
    # prob <- pracma::integral(integrand, lower_bound, upper_bound)
    prob <- integrate(integrand, lower = lower_bound, upper = upper_bound, subdivisions = 500, rel.tol = 1e-2)$value

    proba_results[t] <- prob
    }
  return(proba_results)
}

# ----------------------------------------------------------------------------------------
# APLLY PROBA CALCULATION TO HISTORY LOG
#-----------------------------------------------------------------------------------------

library(dplyr)
library(tidyr)

# Function to compute probabilities for each row inside a group (agent, sim)
compute_probas <- function(df) {
  # Extract contexts for the entire (agent, sim) group (same for all t)
  contexts <- df$context  # Already a list

  # Initialize a list to store probabilities for each row
  probas <- vector("list", nrow(df))

  # Iterate over each time step `t` within this (agent, sim) group
  for (i in seq_len(nrow(df))) {
    # Extract A and b for the specific time step `t`
    A_inv <- df$theta[[i]]$A_inv  # Already a list of matrices
    b <- df$theta[[i]]$b  # Already a list of vectors

    # Extract chosen arm for this row (ind_arm is just the 'choice' column)
    ind_arm <- df$choice[i]

    # Compute probability using the function
    # probas[[i]] <- get_proba_c_eps_greedy(eps = 0.1, A = A, b = b, contexts = contexts, ind_arm = ind_arm)
    probas[[i]] <- get_proba_thompson(A_inv = A_inv, b = b, contexts = contexts, ind_arm = ind_arm)
  }

  # Add probabilities back to the dataframe as a list-column
  df$probas <- probas
  return(df)
}

# Apply function efficiently per (agent, sim) group
res_subset_updated <- res_subset %>%
  group_by(agent, sim) %>%
  group_modify(~ compute_probas(.x)) %>%
  ungroup()

check <- res_subset_updated$probas


# GET PROBA THOMPSON SAMPLING


# Example Usage
theta_hat_example <- list(
  c(1.0, -0.5),  # Theta for arm 1
  c(0.5, 0.2),   # Theta for arm 2
  c(-0.2, 0.8)   # Theta for arm 3
)

sigma_hat_example <- list(
  matrix(c(0.3, 0.1, 0.1, 0.2), 2, 2),  # Covariance for arm 1
  matrix(c(0.4, 0.05, 0.05, 0.3), 2, 2), # Covariance for arm 2
  matrix(c(0.2, 0.02, 0.02, 0.25), 2, 2) # Covariance for arm 3
)

Xa_example <- matrix(c(1, 2,  # Context vector for arm 1
                       0.5, 1,  # Context vector for arm 2
                       -1, 0.5), 3, 2, byrow = TRUE)

chosen_arm <- 1  # Compute probability for arm 1

p_max_single_arm(theta_hat_example, sigma_hat_example, Xa_example, chosen_arm)



# Load required library
library(pracma)

# Generate test data
set.seed(42)
K <- 3  # Number of arms
d <- 2  # Context dimension
nb_timesteps <- 1  # Single test case

# Generate random A_inv (d × d matrices) and b (d × 1 vectors)
A_inv <- lapply(1:K, function(i) solve(diag(runif(d, 0.1, 1))))
b <- lapply(1:K, function(i) matrix(runif(d, -1, 1), nrow = d, ncol = 1))

# Generate random context
contexts <- list(matrix(runif(d, -1, 1), nrow = 1))

# Set an arbitrary chosen arm (matching R's 1-based indexing)
ind_arm <- sample(1:K, 1)

# Run the function
proba_results <- get_proba_thompson(sigma = 0.01, A_inv = A_inv, b = b, contexts = contexts, ind_arm = ind_arm)

# Display results
print(proba_results)



# Load required library
library(pracma)

# Generate test data
set.seed(42)
K <- 3  # Number of arms
d <- 2  # Context dimension
nb_timesteps <- 1  # Single test case

# Generate random A_inv (d × d matrices) and b (d × 1 vectors)
A_inv <- lapply(1:K, function(i) solve(diag(runif(d, 0.1, 1))))
b <- lapply(1:K, function(i) matrix(runif(d, -1, 1), nrow = d, ncol = 1))

# Generate random context
contexts <- list(matrix(runif(d, -1, 1), nrow = 1))

# Set an arbitrary chosen arm (matching R's 1-based indexing)
ind_arm <- sample(1:K, 1)

# Run the function
proba_results <- get_proba_thompson(sigma = 0.01, A_inv = A_inv, b = b, contexts = contexts, ind_arm = ind_arm)

# Monte Carlo Simulation
num_samples <- 100000  # Large number of samples for better accuracy

# Sample from the chosen arm’s distribution
mean_k <- contexts[[1]] %*% b[[ind_arm]]
var_k  <- as.numeric(contexts[[1]] %*% A_inv[[ind_arm]] %*% t(contexts[[1]]))
samples_k <- rnorm(num_samples, mean = mean_k, sd = sqrt(var_k))

# Compute empirical probability using Monte Carlo
proba_mc <- mean(sapply(samples_k, function(x) {
  all(sapply(setdiff(1:K, ind_arm), function(i) {
    mean_i <- contexts[[1]] %*% b[[i]]
    var_i  <- as.numeric(contexts[[1]] %*% A_inv[[i]] %*% t(contexts[[1]]))
    pnorm(x, mean = mean_i, sd = sqrt(var_i))  # Probability all others are smaller
  }))
}))

# Print results
print(paste("Function Probability Estimate:", proba_results))
print(paste("Monte Carlo Probability Estimate:", proba_mc))

# Compare and check if the difference is significant
if (abs(proba_results - proba_mc) > 0.1) {
  warning("Significant discrepancy between Monte Carlo and numerical integration results.")
} else {
  print("Monte Carlo and function results are close. Function appears correct.")
}


f <- function(x) {
  print(x)
  return(sin(x))
}

# Perform integration
integrate(f, 0, pi)

# Check unique x values used
length(unique(track_x))



