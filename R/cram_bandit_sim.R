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
#' @export
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
               Agent$new(CramLinUCBDisjointPolicy$new(0.6, 0.5), bandit, "CramLinUCB"),
               Agent$new(LinUCBDisjointPolicy$new(0.6), bandit, "LinUCB"))

simulation     <- Simulator$new(agents, horizon, simulations,
                                do_parallel = TRUE, save_theta = TRUE,
                                save_context = TRUE)

history        <- simulation$run()

plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")


res <- history$data

param_lookup <- history$data$theta










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

