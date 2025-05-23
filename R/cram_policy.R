#' Cram Policy: Efficient Simultaneous Policy Learning and Evaluation
#'
#' This function performs the cram method (simultaneous policy learning and evaluation)
#' for binary policies on data including covariates (X), binary treatment indicator (D) and outcomes (Y).
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for non-treated).
#' @param Y A vector of outcome values for each sample.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a vector of length equal to the sample size providing the batch assignment (index) for each individual in the sample.
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}. Note: you can also set model_type to NULL and specify custom_fit and custom_predict to use your custom model.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression, \code{"fnn"} for Feedforward Neural Network and \code{"caret"} for Caret. Default is \code{"ridge"}. if model_type is 'causal_forest', choose NULL, if model_type is 's_learner' or 'm_learner', choose between 'ridge', 'fnn' and 'caret'.
#' @param baseline_policy A list providing the baseline policy (binary 0 or 1) for each sample. If \code{NULL}, defaults to a list of zeros with the same length as the number of rows in \code{X}.
#' @param parallelize_batch Logical. Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches. They are learned in parallel when parallelize_batch is TRUE vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training). Defaults to \code{FALSE}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @param custom_fit A custom, user-defined, function that outputs a fitted model given training data (allows flexibility). Defaults to \code{NULL}.
#' @param custom_predict A custom, user-defined, function for making predictions given a fitted model and test data (allow flexibility). Defaults to \code{NULL}.
#' @param alpha Significance level for confidence intervals. Default is 0.05 (95\% confidence).
#' @param propensity The propensity score function for binary treatment indicator (D) (probability for each unit to receive treatment). Defaults to 0.5 (random assignment).
#' @return A list containing:
#' \itemize{
#'   \item \code{raw_results}: A data frame summarizing key metrics with truncated decimals:
#'     \itemize{
#'       \item \code{Delta Estimate}: The estimated treatment effect (delta).
#'       \item \code{Delta Standard Error}: The standard error of the delta estimate.
#'       \item \code{Delta CI Lower}: The lower bound of the confidence interval for delta.
#'       \item \code{Delta CI Upper}: The upper bound of the confidence interval for delta.
#'       \item \code{Policy Value Estimate}: The estimated policy value.
#'       \item \code{Policy Value Standard Error}: The standard error of the policy value estimate.
#'       \item \code{Policy Value CI Lower}: The lower bound of the confidence interval for policy value.
#'       \item \code{Policy Value CI Upper}: The upper bound of the confidence interval for policy value.
#'       \item \code{Proportion Treated}: The proportion of individuals treated under the final policy.
#'     }
#'   \item \code{interactive_table}: An interactive table summarizing key metrics for detailed exploration.
#'   \item \code{final_policy_model}: The final fitted policy model based on \code{model_type} and \code{learner_type} or \code{custom_fit}.
#' }
#'
#' @examples
#' # Example data
#' X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
#' D_data <- as.integer(sample(c(0, 1), 100, replace = TRUE))
#' Y_data <- rnorm(100)
#' nb_batch <- 5
#'
#' # Perform CRAM policy
#' result <- cram_policy(X = X_data,
#'                           D = D_data,
#'                           Y = Y_data,
#'                           batch = nb_batch)
#'
#' # Access results
#' result$raw_results
#' result$interactive_table
#' result$final_policy_model
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{keras_model_sequential}}
#' @importFrom DT datatable
#' @export


# Combined experiment function
cram_policy <- function(X, D, Y, batch, model_type = "causal_forest",
                            learner_type = "ridge", baseline_policy = NULL,
                            parallelize_batch = FALSE, model_params = NULL,
                            custom_fit = NULL, custom_predict = NULL,
                            alpha=0.05, propensity = NULL) {

  ## CRAM LEARNING --------------------------------------------------------------------------
  learning_result <- cram_learning(X, D, Y, batch, model_type = model_type,
                                   learner_type = learner_type, baseline_policy = baseline_policy,
                                   parallelize_batch = parallelize_batch, model_params = model_params,
                                   custom_fit = custom_fit, custom_predict = custom_predict,
                                   propensity = propensity)


  policies <- learning_result$policies
  batch_indices <- learning_result$batch_indices
  final_policy_model <- learning_result$final_policy_model
  nb_batch <- length(batch_indices)

  # ------------------------------------------------------------------------------------------

  ## PROPORTION OF TREATED UNDER FINAL POLICY
  final_policy <- policies[, nb_batch + 1]  # Extract the final policy
  proportion_treated <- mean(final_policy)  # Proportion of treated individuals

  ## POLICY VALUE DIFFERENCE (DELTA) ---------------------------------------------------------
  # estimate
  delta_estimate <- cram_estimator(X, Y, D, policies, batch_indices, propensity = propensity)

  # variance
  delta_asymptotic_variance <- cram_variance_estimator(X, Y, D, policies, batch_indices, propensity = propensity)
  delta_asymptotic_sd <- sqrt(delta_asymptotic_variance)  # v_T, the asymptotic standard deviation
  # delta_standard_error <- delta_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)
  delta_standard_error <- delta_asymptotic_sd

  # confidence interval
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  delta_ci_lower <- delta_estimate - z_value * delta_standard_error
  delta_ci_upper <- delta_estimate + z_value * delta_standard_error
  delta_confidence_interval <- c(delta_ci_lower, delta_ci_upper)

  ## POLICY VALUE (PSI) ---------------------------------------------------------------------
  # estimate
  policy_value_estimate <- cram_policy_value_estimator(X, Y, D,
                                                       policies,
                                                       batch_indices,
                                                       propensity = propensity)

  # variance
  policy_value_asymptotic_variance <- cram_variance_estimator_policy_value(X, Y, D,
                                                                           policies,
                                                                           batch_indices,
                                                                           propensity = propensity)
  policy_value_asymptotic_sd <- sqrt(policy_value_asymptotic_variance)
  # policy_value_standard_error <- policy_value_asymptotic_sd / sqrt(nb_batch)
  policy_value_standard_error <- policy_value_asymptotic_sd

  # confidence interval
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  policy_value_ci_lower <- policy_value_estimate - z_value * policy_value_standard_error
  policy_value_ci_upper <- policy_value_estimate + z_value * policy_value_standard_error
  policy_value_confidence_interval <- c(policy_value_ci_lower, policy_value_ci_upper)



  ## RESULTS: SUMMARY TABLES ----------------------------------------------------------------
  summary_table <- data.frame(
    Metric = c("Delta Estimate", "Delta Standard Error", "Delta CI Lower", "Delta CI Upper",
               "Policy Value Estimate", "Policy Value Standard Error", "Policy Value CI Lower", "Policy Value CI Upper",
               "Proportion Treated"),
    Value = round(c(delta_estimate, delta_standard_error, delta_ci_lower, delta_ci_upper,
                    policy_value_estimate, policy_value_standard_error, policy_value_ci_lower, policy_value_ci_upper,
                    proportion_treated), 5)  # Truncate to 5 decimals
  )

  # Create an interactive table
  interactive_table <- datatable(
    summary_table,
    options = list(pageLength = 5),  # 5 rows, no extra controls
    caption = "CRAM Experiment Results"
  )


  # Return results as a list with raw data, styled outputs, and the model
  return(list(
    raw_results = summary_table,      # Raw table data for programmatic use
    interactive_table = interactive_table, # Interactive table for exploration
    final_policy_model = final_policy_model  # Model (not displayed in the summary)
  ))

}
