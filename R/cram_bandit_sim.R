utils::globalVariables(c(
  "context", "theta_na", "theta", "sim", "num_nulls", "agent",
  "choice", "reward", "probas", "arms", "rewards", "estimate",
  "variance_est", "std_error"
))


#' On-policy CRAM Bandit
#'
#' This function performs simulation for cram bandit policy learning and evaluation.
#'
#' @param horizon The number of timesteps
#' @param simulations The number of simulations
#' @param bandit The bandit, generating contextual features and observed rewards
#' @param policy The policy, choosing the arm at each timestep
#' @param alpha Significance level for confidence intervals for calculating the empirical coverage. Default is 0.05 (95\% confidence).
#' @param do_parallel Whether to parallelize the simulations. Default to FALSE.
#'
#' @return A **list** containing:
#'   \item{raw_results}{A data frame summarizing key metrics:
#'   Average Prediction Error on the Policy Value Estimate,
#'   Average Prediction Error on the Variance Estimate,
#'   Empirical Coverage of the alpha level Confidence Interval.}
#'   \item{interactive_table}{An interactive table summarizing key metrics for detailed exploration.}
#'
#' @import contextual
#' @importFrom magrittr %>%
#' @import data.table
#' @importFrom dplyr group_by group_modify ungroup summarise mutate select
#' @importFrom purrr map2_dbl
#' @importFrom stats glm predict qnorm rbinom rnorm var dnorm pnorm integrate
#' @importFrom contextual ContextualLinearBandit ContextualEpsilonGreedyPolicy
#' @export


cram_bandit_sim <- function(horizon, simulations,
                            bandit, policy,
                            alpha=0.05, do_parallel = FALSE) {

  horizon <- as.integer(horizon)
  # Add 1 as first sim of the contextual package has a writing error for theta
  simulations <- as.integer(simulations + 1)

  policy_name <- policy$class_name

  if (is.null(policy$batch_size)) {
    batch_size <- 1
  } else {
    batch_size <- policy$batch_size
  }

  agents <- list(Agent$new(policy, bandit, policy_name))

  simulation     <- Simulator$new(agents, horizon, simulations,
                                  do_parallel = do_parallel,
                                  save_theta = TRUE,
                                  save_context = TRUE)

  history        <- simulation$run()

  plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")

  res <- history$data

  list_betas <<- list_betas[-1]  # Remove the first element

  # Retrieve d (feature dimension) from any row, assuming d is constant across rows
  d_value <- res$d[1L]  # Using 1L for integer indexing optimization

  # Dynamically select the X.1 to X.d columns
  X_columns <- paste0("X.", seq_len(d_value))

  # Subset res, keeping only the relevant columns
  res_subset <- res[, c("agent", "sim", "t", "choice", "reward", "theta", X_columns), with = FALSE]
  # check <- copy(res_subset)
  # Convert X.1, ..., X.d into a single list-column `context`
  res_subset[, context := asplit(as.matrix(.SD), 1), .SDcols = X_columns]

  # Convert X.1, ..., X.d into a single list-column `context`
  # res_subset[, context := lapply(1:.N, function(i) unlist(.SD[i, ], use.names = FALSE)), .SDcols = X_columns]

  # Remove original X.1, ..., X.d columns after storing in `context`
  res_subset[, (X_columns) := NULL]


  # # Retrieve k (number of arms) dynamically from any row
  # k_value <- res$k[1]
  #
  # # Convert context vectors into k × d matrices
  # res_subset[, context := lapply(context, function(vec) matrix(rep(vec, each = k_value), nrow = k_value, byrow = TRUE))]


  # Check for NULL to ensure the history table is clean

  # Count NULL values in theta per simulation (direct NULL checking)
  sims_to_remove <- res_subset[, .(num_nulls = sum(vapply(theta, is.null, logical(1L)))), by = sim][num_nulls > 0, sim]

  # Remove simulations where num_nulls >= 1 (optimized filtering)
  if (length(sims_to_remove) > 0) {
    res_subset <- res_subset[!sim %in% sims_to_remove]
  }

  # # Convert NULL values in theta to NA
  # res_subset[, theta_na := lapply(theta, function(x) if (is.null(x)) NA else x)]
  #
  # # Count the number of NA values in theta_na per simulation
  # null_counts <- res_subset[, .(num_nulls = sum(is.na(theta_na))), by = sim]
  #
  # # Identify simulations to drop (where num_nulls >= 1)
  # sims_to_remove <- null_counts[num_nulls >= 1, sim]
  #
  # # Remove these simulations from res_subset in place
  # res_subset <- res_subset[!sim %in% sims_to_remove]
  #
  # # Drop the temporary column
  # res_subset[, theta_na := NULL]

  # Ensure data is sorted correctly by agent, sim, t
  setorder(res_subset, agent, sim, t)

  res_subset <- res_subset %>%
    mutate(sim = sim - 1)

  # Apply function efficiently per (agent, sim) group
  # res_subset_updated <- res_subset %>%
  #   group_by(agent, sim) %>%
  #   group_modify(~ compute_probas(.x, policy, policy_name)) %>%
  #   ungroup()

  # other <- copy(res_subset)
  res_subset_updated <- res_subset[, compute_probas(.SD, policy, policy_name, batch_size = batch_size), by = .(agent, sim)]
  # res_subset_updated <- res_subset[, compute_probas(.SD, policy, policy_name), by = .(agent, sim)]


  # return(list(a = res_subset_updated, b = other))


  # Process Data by Simulation
  estimates <- res_subset_updated %>%
    group_by(sim) %>%
    summarise(
      arms = list(choice),
      rewards = list(reward),
      policy_matrix = list(do.call(cbind, probas))  # Concatenate probability vectors into a matrix
    ) %>%
    mutate(
      estimate = map2_dbl(arms, rewards, ~ cram_bandit_est(policy_matrix[[cur_group_id()]], .y, .x, batch=batch_size)),
      variance_est = map2_dbl(arms, rewards, ~ cram_bandit_var(policy_matrix[[cur_group_id()]], .y, .x, batch=batch_size)),
      estimand = map_dbl(sim, ~ compute_estimand(.x, res_subset_updated, list_betas, eps = 0.1))
    )


  # # Process Data by Simulation
  # estimates <- res_subset_updated %>%
  #   group_by(sim) %>%
  #   summarise(
  #     arms = list(choice),  # Vector of chosen arms
  #     rewards = list(reward),  # Vector of rewards
  #     policy_matrix = list(do.call(cbind, probas))  # Concatenate probability vectors into a matrix
  #   ) %>%
  #   mutate(
  #     estimate = map2_dbl(arms, rewards, ~ cram_bandit_est(policy_matrix[[cur_group_id()]], .y, .x, batch=batch_size)),
  #     variance_est = map2_dbl(arms, rewards, ~ cram_bandit_var(policy_matrix[[cur_group_id()]], .y, .x, batch=batch_size))  # Variance estimation
  #   )

  # Compute prediction error
  estimates <- estimates %>%
    mutate(prediction_error = estimate - estimand)

  # Compute empirical bias (mean of prediction error)
  empirical_bias <- mean(estimates$prediction_error)

  estimates <- estimates %>%
    mutate(est_rel_error = (estimate - estimand) / estimand)

  # Compute True Variance (Sample Variance of Prediction Errors)
  true_variance <- var(estimates$prediction_error)

  print("True variance")
  print(true_variance)

  # Compute Prediction Error on Variance
  estimates <- estimates %>%
    mutate(variance_prediction_error = (variance_est - true_variance) / true_variance)

  # Exclude 20% of Simulations Randomly
  num_excluded <- ceiling(0.2 * nrow(estimates))  # 20% of total sims
  excluded_sims <- sample(nrow(estimates), size = num_excluded)

  # Select only the remaining 80% of simulations
  filtered_errors <- estimates$est_rel_error[-excluded_sims]
  filtered_variance_errors <- estimates$variance_prediction_error[-excluded_sims]

  # Compute and Report Final Average Prediction Errors
  avg_prediction_error <- mean(filtered_errors)
  avg_variance_prediction_error <- mean(filtered_variance_errors)

  print(paste("Average Prediction Error:", avg_prediction_error))
  print(paste("Average Variance Prediction Error:", avg_variance_prediction_error))

  # Compute 95% Confidence Intervals
  z_value <- qnorm(1 - alpha / 2)
  T_steps <- max(res_subset_updated$t)  # Get total timesteps
  estimates <- estimates %>%
    mutate(
      # std_error = sqrt(variance_est) * sqrt(T_steps - 1),
      std_error = sqrt(variance_est),
      ci_lower = estimate - z_value * std_error,
      ci_upper = estimate + z_value * std_error
    )

  # Compute Empirical Coverage
  empirical_coverage <- mean((estimates$estimand >= estimates$ci_lower) &
                               (estimates$estimand <= estimates$ci_upper))

  print(paste("Empirical Coverage of 95% Confidence Interval:", empirical_coverage))

}
