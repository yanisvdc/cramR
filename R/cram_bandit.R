#' Cram Bandit: On-policy Statistical Evaluation in Contextual Bandits
#'
#' Performs the Cram method for On-policy Statistical Evaluation
#' in Contextual Bandits
#'
#' @param pi An array of shape (T × B, T, K) or (T × B, T),
#' where T is the number of learning steps (or policy updates),
#' B is the batch size, K is the number of arms,
#' T x B is the total number of contexts.
#' If 3D, pi[j, t, a] gives the probability that
#' the policy pi_t assigns arm a to context X_j.
#' If 2D, pi[j, t] gives the probability that the policy pi_t
#' assigns arm A_j (arm actually chosen under X_j in the history)
#' to context X_j. Please see vignette for more details.
#' @param arm A vector of length T x B indicating which arm was selected in each context
#' @param reward A vector of observed rewards of length T x B
#' @param batch (Optional) A vector or integer. If a vector, gives the
#' batch assignment for each context. If an integer, interpreted as the batch
#' size and contexts are assigned to a batch in the order of the dataset.
#' Default is 1.
#' @param alpha Significance level for confidence intervals for calculating the empirical coverage.
#' Default is 0.05 (95\% confidence).
#'
#' @return A list containing:
#'   \item{raw_results}{A data frame summarizing key metrics:
#'   Empirical Bias on Policy Value,
#'   Average relative error on Policy Value,
#'   RMSE using relative errors on Policy Value,
#'   Empirical Coverage of Confidence Intervals.}
#'   \item{interactive_table}{An interactive table summarizing the same key metrics in a user-friendly interface.}
#'
#' @examples
#' # Example with batch size of 1
#'
#' # Set random seed for reproducibility
#' set.seed(42)
#'
#' # Define parameters
#' T <- 100  # Number of timesteps
#' K <- 4    # Number of arms
#'
#' # Simulate a 3D array pi of shape (T, T, K)
#' # - First dimension: Individuals (context Xj)
#' # - Second dimension: Time steps (pi_t)
#' # - Third dimension: Arms (depth)
#' pi <- array(runif(T * T * K, 0.1, 1), dim = c(T, T, K))
#'
#' # Normalize probabilities so that each row sums to 1 across arms
#' for (t in 1:T) {
#'   for (j in 1:T) {
#'     pi[j, t, ] <- pi[j, t, ] / sum(pi[j, t, ])
#'     }
#'  }
#'
#' # Simulate arm selections (randomly choosing an arm)
#' arm <- sample(1:K, T, replace = TRUE)
#'
#' # Simulate rewards (assume normally distributed rewards)
#' reward <- rnorm(T, mean = 1, sd = 0.5)
#'
#' result <- cram_bandit(pi, arm, reward, batch=1, alpha=0.05)
#' result$raw_results
#'
#' @importFrom caret train
#' @importFrom DT datatable
#' @importFrom stats qnorm
#' @importFrom parallel detectCores
#' @export

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
