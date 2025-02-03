#' CRAM ML
#'
#' This function performs the cram method (simultaneous ML learning and evaluation)
#' on experimental or observational data, for which the data generation process is unknown.
#'
#' @param data A matrix or data frame of covariates for each sample.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a vector of length equal to the sample size providing the batch assignment (index) for each individual in the sample.
#' @param formula Optional formula relating the target to the predictors for supervised learning (NULL for unsupervised)
#' @param caret_params The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param parallelize_batch Logical. Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches. They are learned in parallel when parallelize_batch is TRUE vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training). Defaults to \code{FALSE}.
#' @param custom_fit A custom, user-defined, function that outputs a fitted model given training data (allows flexibility). Defaults to \code{NULL}.
#' @param custom_predict A custom, user-defined, function for making predictions given a fitted model and test data (allow flexibility). Defaults to \code{NULL}.
#' @param custom_loss A custom loss
#' @param alpha Significance level for confidence intervals. Default is 0.05 (95\% confidence).
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
#' D_data <- D_data <- as.integer(sample(c(0, 1), 100, replace = TRUE))
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
cram_ml <- function(data, batch, formula=NULL, caret_params = NULL,
                    parallelize_batch = FALSE, loss_name='accuracy',
                    custom_fit = NULL, custom_predict = NULL,
                    custom_loss = NULL, alpha=0.05) {

  ## CRAM LEARNING --------------------------------------------------------------------------
  learning_result <- ml_learning(data=data, formula=formula, batch=batch,
                                 parallelize_batch = parallelize_batch,
                                 loss_name = loss_name,
                                 caret_params = caret_params,
                                 custom_fit = custom_fit,
                                 custom_predict = custom_predict,
                                 custom_loss = custom_loss,
                                 n_cores = detectCores() - 1)


  losses <- learning_result$losses
  batch_indices <- learning_result$batch_indices
  final_ml_model <- learning_result$final_ml_model
  nb_batch <- length(batch_indices)

  # ------------------------------------------------------------------------------------------

  ## EXPECTED LOSS  --------------------------------------------------------------------------
  # estimate
  expected_loss_estimate <- cram_expected_loss(losses, batch_indices)

  # variance
  expected_loss_asymptotic_variance <- cram_var_expected_loss(losses, batch_indices)
  expected_loss_asymptotic_sd <- sqrt(expected_loss_asymptotic_variance)  # v_T, the asymptotic standard deviation
  expected_loss_standard_error <- expected_loss_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

  # confidence interval
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  expected_loss_ci_lower <- expected_loss_estimate - z_value * expected_loss_standard_error
  expected_loss_ci_upper <- expected_loss_estimate + z_value * expected_loss_standard_error
  expected_loss_confidence_interval <- c(expected_loss_ci_lower, expected_loss_ci_upper)


  ## RESULTS: SUMMARY TABLES ----------------------------------------------------------------
  summary_table <- data.frame(
    Metric = c("Expected Loss Estimate", "Expected Loss Standard Error",
               "Expected Loss CI Lower", "Expected Loss CI Upper"),
    Value = round(c(expected_loss_estimate, expected_loss_standard_error,
                    expected_loss_ci_lower, expected_loss_ci_upper), 5)  # Truncate to 5 decimals
  )

  # Create an interactive table
  interactive_table <- datatable(
    summary_table,
    options = list(pageLength = 5),  # 5 rows, no extra controls
    caption = "CRAM ML Results"
  )


  # Return results as a list with raw data, styled outputs, and the model
  return(list(
    raw_results = summary_table,      # Raw table data for programmatic use
    interactive_table = interactive_table, # Interactive table for exploration
    final_ml_model = final_ml_model  # Model (not displayed in the summary)
  ))

}
