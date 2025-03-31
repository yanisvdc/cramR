utils::globalVariables(c(
  "context", "theta_na", "theta", "sim", "num_nulls", "agent",
  "choice", "reward", "probas", "arms", "rewards", "estimate",
  "variance_est", "std_error", "list_betas", "prediction_error", "estimand", "est_rel_error",
  "variance_prediction_error", "ci_lower", "ci_upper"
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
#' @param seed Seed for simulation
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
                            alpha=0.05, do_parallel = FALSE, seed=42) {

  # Force garbage collection before starting
  gc(full = TRUE, verbose = FALSE)

  # RUN SIMULATION ---------------------------------------------------------

  horizon <- as.integer(horizon)
  # Add 1 as first sim of the contextual package has a writing error for theta
  simulations <- as.integer(simulations + 1)

  policy_name <- policy$class_name

  # list_betas <<- NULL  # Prevent R CMD check NOTE on <<- assignment

  list_betas <<- list()

  if (is.null(policy$batch_size)) {
    batch_size <- 1
  } else {
    batch_size <- policy$batch_size
  }

  agents <- list(Agent$new(policy, bandit, policy_name))

  simulation     <- Simulator$new(agents, horizon, simulations,
                                  do_parallel = do_parallel,
                                  save_theta = TRUE,
                                  save_context = TRUE, set_seed = seed)

  history        <- simulation$run()

  plot(history, type = "cumulative", rate = FALSE, legend_position = "topleft")

  res <- history$data

  # PROCESS HISTORY TABLE -------------------------------------------------------

  # Convert to data.table without copy
  setDT(res)

  list_betas <<- list_betas[-1]  # Remove the first element as first sim has writing error
  d_value <- res$d[1L]

  # Context is given through d columns: X.1 to X.d
  X_columns <- paste0("X.", seq_len(d_value))

  res <- res[, c("agent", "sim", "t", "choice", "reward", "theta", X_columns), with = FALSE]

  # Convert X.1, ..., X.d into a single list-column `context` and remove old columns
  res[, context := asplit(as.matrix(.SD), 1), .SDcols = X_columns]
  res[, (X_columns) := NULL]

  # Count NULL values in theta per simulation
  sims_to_remove <- res[, .(num_nulls = sum(vapply(theta, is.null, logical(1L)))), by = sim][num_nulls > 0, sim]

  # Remove simulations where num_nulls >= 1
  if (length(sims_to_remove) > 0) {
    res <- res[!sim %in% sims_to_remove]
  }

  # Ensure data is sorted correctly by agent, sim, t
  setorder(res, agent, sim, t)

  # As we remove first sim due to writing error, shift sim numbers by -1
  # res <- res %>%
  #   mutate(sim = sim - 1)
  res[, sim := sim - 1L]


  # CALCULATE PROBAS ---------------------------------------------------------------

  # res <- res[, compute_probas(.SD, policy, policy_name, batch_size = batch_size), by = .(agent, sim)]

  # res[, probas := compute_probas(.SD, policy, policy_name, batch_size = batch_size)$probas, by = .(agent, sim)]

  # res[, probas := vector("list", .N)]  # Pre-allocate a list column with NULLs
  # res[, probas := compute_probas(.SD, policy, policy_name, batch_size = batch_size), by = .(agent, sim)]

  # # Pre-allocate probas column as a list to avoid memory spikes
  # res[, probas := vector("list", .N)]
  #
  # # Compute probabilities with NA instead of NULL
  # res[, probas := compute_probas(.SD, policy, policy_name, batch_size = batch_size), by = .(agent, sim)]
  #
  # # Convert NA values back to NULL after assignment
  # res[, probas := lapply(probas, function(x) if (is.na(x)) NULL else x)]

  res[, probas := list(compute_probas(.SD, policy, policy_name, batch_size = batch_size)),
      by = .(agent, sim)]

  # # Memory-optimized replacement:
  # res[, probas := {
  #   # Compute probabilities for this group
  #   tmp <- compute_probas(.SD, policy, policy_name, batch_size = batch_size)
  #
  #   # Return JUST the probas column content for this group
  #   list(tmp$probas)
  # }, by = .(agent, sim)]

  print("Check compute probas")

  # CALCULATE STATISTICS -----------------------------------------------------------

  print(Sys.time())

  print("Betas")
  print(list_betas)

  # Compute estimates using data.table
  estimates <- res[, {
    # Extract arms (choices) and rewards as vectors
    arms_vec <- choice
    rewards_vec <- reward

    # Build policy matrix: remove NULL probas and column-bind
    policy_mat <- do.call(rbind, probas)

    # Compute estimates and variance using policy matrix
    est <- cram_bandit_est(policy_mat, rewards_vec, arms_vec, batch = batch_size)
    var_est <- cram_bandit_var(policy_mat, rewards_vec, arms_vec, batch = batch_size)

    # Compute estimand using the entire group's data
    estimand_val <- compute_estimand(.SD, list_betas, policy, policy_name, batch_size, bandit)

    # Return results as columns
    .(estimate = est, variance_est = var_est, estimand = estimand_val)
  }, by = sim]

  print("Check estimates")

  # Compute prediction error and other metrics using data.table
  estimates[, prediction_error := estimate - estimand]

  # Compute empirical bias (mean of prediction error)
  empirical_bias <- estimates[, mean(prediction_error)]

  # Add relative error column
  estimates[, est_rel_error := (estimate - estimand) / estimand]

  # Compute True Variance (Sample Variance of Prediction Errors)
  true_variance <- estimates[, var(prediction_error)]

  print("True variance")
  print(true_variance)

  # Add variance prediction error column
  estimates[, variance_prediction_error := (variance_est - true_variance) / true_variance]

  # Exclude 20% of Simulations Randomly
  num_excluded <- ceiling(0.2 * nrow(estimates))
  excluded_sims <- sample(nrow(estimates), size = num_excluded)

  # Filter errors using data.table's anti-join syntax
  filtered_dt <- estimates[!excluded_sims]

  # Compute final average prediction errors
  avg_prediction_error <- filtered_dt[, mean(est_rel_error)]
  avg_variance_prediction_error <- filtered_dt[, mean(variance_prediction_error)]

  print(paste("Average Prediction Error:", avg_prediction_error))
  print(paste("Average Variance Prediction Error:", avg_variance_prediction_error))

  # Compute 95% Confidence Intervals
  z_value <- qnorm(1 - alpha / 2)

  estimates[, `:=`(
    std_error = sqrt(variance_est),
    ci_lower = estimate - z_value * sqrt(variance_est),
    ci_upper = estimate + z_value * sqrt(variance_est)
  )]

  # Compute Empirical Coverage
  empirical_coverage <- estimates[, mean(estimand >= ci_lower & estimand <= ci_upper)]

  print(paste("Empirical Coverage of 95% Confidence Interval:", empirical_coverage))
  print(Sys.time())

  # Final cleanup before return
  gc(full = TRUE, verbose = FALSE)

  return(estimates)

}
