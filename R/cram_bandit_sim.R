# Load necessary libraries
library(dplyr)
library(purrr)  # For `map2_dbl`

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

cram_bandit_sim <- function(horizon, simulations,
                            bandit, policy, policy_name, do_parallel = FALSE) {

  horizon <- as.integer(horizon)
  # Add 1 as first sim of the contextual package has a writing error for theta
  simulations <- as.integer(simulations + 1)

  agents <- list(Agent$new(policy, bandit, policy_name))

  simulation     <- Simulator$new(agents, horizon, simulations,
                                  do_parallel = do_parallel,
                                  save_theta = TRUE,
                                  save_context = TRUE)

  history        <- simulation$run()

  plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")

  res <- history$data

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


  # Check for NULL to ensure the history table is clean

  # Convert NULL values in theta to NA
  res_subset[, theta_na := lapply(theta, function(x) if (is.null(x)) NA else x)]

  # Count the number of NA values in theta_na per simulation
  null_counts <- res_subset[, .(num_nulls = sum(is.na(theta_na))), by = sim]

  # Identify simulations to drop (where num_nulls >= 1)
  sims_to_remove <- null_counts[num_nulls >= 1, sim]

  # Remove these simulations from res_subset in place
  res_subset <- res_subset[!sim %in% sims_to_remove]

  # Drop the temporary column
  res_subset[, theta_na := NULL]

  # Ensure data is sorted correctly by agent, sim, t
  setorder(res_subset, agent, sim, t)

  # Apply function efficiently per (agent, sim) group
  res_subset_updated <- res_subset %>%
    group_by(agent, sim) %>%
    group_modify(~ compute_probas(.x)) %>%
    ungroup()

  check <- res_subset_updated$probas

  # Process Data by Simulation
  estimates <- res_subset_updated %>%
    group_by(sim) %>%
    summarise(
      arms = list(choice),  # Vector of chosen arms
      rewards = list(reward),  # Vector of rewards
      policy_matrix = list(do.call(cbind, probas))  # Concatenate probability vectors into a matrix
    ) %>%
    mutate(
      estimate = map2_dbl(arms, rewards, ~ cram_bandit_est(policy_matrix[[cur_group_id()]], .y, .x)),
      variance_est = map2_dbl(arms, rewards, ~ cram_bandit_var(policy_matrix[[cur_group_id()]], .y, .x))  # Variance estimation
    )

  # Compute True Estimate (Average across Sims)
  true_estimate <- mean(estimates$estimate)

  # Compute Prediction Errors
  estimates <- estimates %>%
    mutate(prediction_error = estimate - true_estimate)

  # Compute True Variance (Sample Variance of Prediction Errors)
  true_variance <- var(estimates$prediction_error)

  # Compute Prediction Error on Variance
  estimates <- estimates %>%
    mutate(variance_prediction_error = variance_est - true_variance)

  # Exclude 20% of Simulations Randomly
  set.seed(123)  # Ensure reproducibility
  num_excluded <- ceiling(0.2 * nrow(estimates))  # 20% of total sims
  excluded_sims <- sample(nrow(estimates), size = num_excluded)

  # Select only the remaining 80% of simulations
  filtered_errors <- estimates$prediction_error[-excluded_sims]
  filtered_variance_errors <- estimates$variance_prediction_error[-excluded_sims]

  # Compute and Report Final Average Prediction Errors
  avg_prediction_error <- mean(filtered_errors)
  avg_variance_prediction_error <- mean(filtered_variance_errors)

  print(paste("Average Prediction Error:", avg_prediction_error))
  print(paste("Average Variance Prediction Error:", avg_variance_prediction_error))

  # Compute 95% Confidence Intervals
  T_steps <- max(res_subset_updated$t)  # Get total timesteps
  estimates <- estimates %>%
    mutate(
      # std_error = sqrt(variance_est) * sqrt(T_steps - 1),
      std_error = sqrt(variance_est),
      ci_lower = estimate - 1.96 * std_error,
      ci_upper = estimate + 1.96 * std_error
    )

  # Compute Empirical Coverage
  empirical_coverage <- mean((true_estimate >= estimates$ci_lower) & (true_estimate <= estimates$ci_upper))

  print(paste("Empirical Coverage of 95% Confidence Interval:", empirical_coverage))
}
