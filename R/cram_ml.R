#' Cram ML: Simultaneous Machine Learning and Evaluation
#'
#' Performs the Cram method for simultaneous machine learning and evaluation.
#'
#' @param data A matrix or data frame of covariates. For supervised learning,
#'   must include the target variable specified in formula.
#' @param batch Integer specifying number of batches or vector of pre-defined
#'   batch assignments.
#' @param formula Formula for supervised learning (e.g., y ~ .).
#' @param caret_params List of parameters for caret::train() containing:
#' \itemize{
#'   \item{method: Model type (e.g., "rf", "glm", "xgbTree" for supervised learning)}
#'   \item{Additional method-specific parameters}
#' }
#' @param parallelize_batch Logical indicating whether to parallelize batch
#'   processing (default = FALSE).
#' @param custom_fit Optional custom model training function.
#' @param custom_predict Optional custom prediction function.
#' @param custom_loss Optional custom loss function.
#' @param loss_name Name of loss metric (supported: "se", "logloss",
#'   "accuracy").
#' @param alpha Confidence level for intervals (default = 0.05).
#' @param classify Indicate if this is a classification problem. Defaults to FALSE.
#'
#' @return A list containing:
#' \itemize{
#'   \item{raw_results: Data frame with performance metrics}
#'   \item{interactive_table: The same performance metrics in a user-friendly interface}
#'   \item{final_ml_model: Trained model object}
#' }
#'
#' @examples
#' # Load necessary libraries
#' library(caret)
#'
#' # Set seed for reproducibility
#' set.seed(42)
#'
#' # Generate example dataset
#' X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
#' Y_data <- rnorm(100)  # Continuous target variable for regression
#' data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included
#'
#' # Define caret parameters for simple linear regression (no cross-validation)
#' caret_params_lm <- list(
#'   method = "lm",
#'   trControl = trainControl(method = "none")
#' )
#'
#' nb_batch <- 5
#'
#' # Run ML learning function
#' result <- cram_ml(
#'   data = data_df,
#'   formula = Y ~ .,  # Linear regression model
#'   batch = nb_batch,
#'   loss_name = 'se',
#'   caret_params = caret_params_lm
#' )
#'
#' result$raw_results
#' result$interactive_table
#' result$final_ml_model
#' @seealso
#' \code{\link[caret]{train}} for model training parameters
#'
#' @importFrom caret train
#' @importFrom DT datatable
#' @importFrom stats qnorm
#' @importFrom parallel detectCores
#' @export



# Combined experiment function
cram_ml <- function(data, batch, formula=NULL, caret_params = NULL,
                    parallelize_batch = FALSE, loss_name=NULL,
                    custom_fit = NULL, custom_predict = NULL,
                    custom_loss = NULL, alpha=0.05, classify=FALSE) {

  ## CRAM LEARNING --------------------------------------------------------------------------
  learning_result <- ml_learning(data=data, formula=formula, batch=batch,
                                 parallelize_batch = parallelize_batch,
                                 loss_name = loss_name,
                                 caret_params = caret_params,
                                 custom_fit = custom_fit,
                                 custom_predict = custom_predict,
                                 custom_loss = custom_loss,
                                 n_cores = detectCores() - 1,
                                 classify = classify)


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
  # expected_loss_standard_error <- expected_loss_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)
  expected_loss_standard_error <- expected_loss_asymptotic_sd

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
