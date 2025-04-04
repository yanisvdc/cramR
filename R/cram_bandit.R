#' CRAM Bandit - On-policy Statistical Evaluation in Contextual Bandit
#'
#' Performs the Cram method for On-policy Statistical Evaluation
#' in Contextual Bandit
#'
#' @param pi for each row j, column t, depth a, gives pi_t(Xj, a)
#' @param arm arms selected at each time step
#' @param reward rewards at each time step
#' @param batch Batch size
#' @param alpha Confidence level for intervals (default = 0.05).
#'
#' @return A list containing:
#' \itemize{
#'   \item{raw_results: Data frame with performance metrics}
#'   \item{interactive_table: DT::datatable interactive view}
#'   \item{final_ml_model: Trained model object}
#' }
#' @seealso
#' \code{\link[caret]{train}} for model training parameters
#' \code{\link[stats]{kmeans}} for unsupervised clustering
#' \code{\link[DT]{datatable}} for interactive tables
#'
#' @importFrom caret train
#' @importFrom DT datatable
#' @importFrom stats qnorm
#' @importFrom parallel detectCores
#' @export



# Combined experiment function
cram_bandit <- function(pi, arm, reward, batch=1, alpha=0.05) {

  nb_timesteps <- length(arm)

  ## POLICY VALUE   --------------------------------------------------------------------------
  # estimate
  policy_val_estimate <- cram_bandit_est(pi, reward, arm, batch=batch)

  # variance
  policy_val_variance <- cram_bandit_var(pi, reward, arm, batch=batch)
  policy_val_sd <- sqrt(policy_val_variance)
  policy_val_standard_error <- policy_val_sd / sqrt(nb_timesteps - 1)

  # confidence interval
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  policy_val_ci_lower <- policy_val_estimate - z_value * policy_val_standard_error
  policy_val_ci_upper <- policy_val_estimate + z_value * policy_val_standard_error
  policy_val_confidence_interval <- c(policy_val_ci_lower, policy_val_ci_upper)


  ## RESULTS: SUMMARY TABLES ----------------------------------------------------------------
  summary_table <- data.frame(
    Metric = c("Policy Value Estimate", "Policy Value Standard Error",
               "Policy Value CI Lower", "Policy Value CI Upper"),
    Value = round(c(policy_val_estimate, policy_val_standard_error,
                    policy_val_ci_lower, policy_val_ci_upper), 5)  # Truncate to 5 decimals
  )

  # Create an interactive table
  interactive_table <- datatable(
    summary_table,
    options = list(pageLength = 5),  # 5 rows, no extra controls
    caption = "CRAM Bandit Policy Evaluation Results"
  )


  # Return results as a list with raw data, styled outputs, and the model
  return(list(
    raw_results = summary_table,      # Raw table data for programmatic use
    interactive_table = interactive_table # Interactive table for exploration
  ))

}
