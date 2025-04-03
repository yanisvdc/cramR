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

  list_betas <- bandit$list_betas

  # PROCESS HISTORY TABLE -------------------------------------------------------

  # Convert to data.table without copy
  setDT(res)

  list_betas <- list_betas[-1]  # Remove the first element as first sim has writing error
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
  res[, sim := sim - 1L]


  # CALCULATE PROBAS ---------------------------------------------------------------

  res[, probas := list(compute_probas(.SD, policy, policy_name, batch_size = batch_size)),
      by = .(agent, sim)]

  # CALCULATE STATISTICS -----------------------------------------------------------

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

  # Compute prediction error and other metrics using data.table
  estimates[, prediction_error := estimate - estimand]

  # Compute empirical bias (mean of prediction error)
  empirical_bias <- estimates[, mean(prediction_error)]

  # Add relative error column
  estimates[, est_rel_error := (estimate - estimand) / estimand]

  # Rel BIAS
  rel_empirical_bias <- estimates[, mean(est_rel_error)]

  # RMSE
  relative_rmse <- estimates[, sqrt(mean(est_rel_error^2))]

  # Compute True Variance (Sample Variance of Prediction Errors)
  true_variance <- estimates[, var(prediction_error)]

  # Add variance prediction error column
  estimates[, variance_prediction_error := (variance_est - true_variance) / true_variance]

  # Compute 95% Confidence Intervals
  z_value <- qnorm(1 - alpha / 2)

  estimates[, `:=`(
    std_error = sqrt(variance_est),
    ci_lower = estimate - z_value * sqrt(variance_est),
    ci_upper = estimate + z_value * sqrt(variance_est)
  )]

  # Compute Empirical Coverage
  empirical_coverage <- estimates[, mean(estimand >= ci_lower & estimand <= ci_upper)]

  # Final cleanup before return
  gc(full = TRUE, verbose = FALSE)

  # SUMMARY TABLES ---------------------------------------------------------------
  summary_table <- data.frame(
    Metric = c("Empirical Bias on Policy Value",
               "Average relative error on Policy Value",
               "RMSE of errors on Policy Value",
               "Empirical Coverage of Confidence Intervals"),
    Value = round(c(empirical_bias, rel_empirical_bias, relative_rmse, empirical_coverage), 5)
  )

  interactive_table <- DT::datatable(
    summary_table,
    options = list(pageLength = 5),
    caption = "Cram Bandit Simulation: Policy Evaluation Metrics"
  )

  # PLOT -------------------------------------------------------------------------
  # bias_plot <- ggplot(estimates, aes(x = sim, y = 100 * est_rel_error)) +
  #   geom_point(shape = 1, color = "blue", size = 2, stroke = 1) +
  #   geom_hline(yintercept = 0, linetype = "dashed", size = 1) +
  #   labs(
  #     x = "Simulation setup ID",
  #     y = "Bias as percentage of policy value",
  #     title = "Bias per Simulation Setup (Relative Error in %)"
  #   ) +
  #   theme_minimal(base_size = 13)

  # RETURN FULL RESULTS ----------------------------------------------------------
  return(list(
    estimates = estimates,
    summary_table = summary_table,
    interactive_table = interactive_table
  ))

}
