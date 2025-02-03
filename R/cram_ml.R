#' CRAM ML - Simultaneous Machine Learning and Evaluation
#'
#' Performs the CRAM method (Causal Regularization via Approximate Models) for
#' simultaneous machine learning and evaluation in experimental or observational
#' studies with unknown data generating processes.
#'
#' @param data A matrix or data frame of covariates. For supervised learning,
#'   must include the target variable specified in `formula`.
#' @param batch Integer specifying number of batches or vector of pre-defined
#'   batch assignments.
#' @param formula Optional formula for supervised learning (e.g., y ~ .).
#'   Use NULL for unsupervised methods like clustering.
#' @param caret_params List of parameters for `caret::train()` containing:
#' \itemize{
#'   \item{method: Model type (e.g., "rf", "glm", "xgbTree" for supervised;
#'         "kmeans" for clustering)}
#'   \item{Additional method-specific parameters}
#' }
#' @param parallelize_batch Logical indicating whether to parallelize batch
#'   processing (default = FALSE).
#' @param custom_fit Optional custom model training function.
#' @param custom_predict Optional custom prediction function.
#' @param custom_loss Optional custom loss function.
#' @param loss_name Name of loss metric (supported: "mse", "logloss",
#'   "accuracy", "euclidean_distance", "pca_projection_error").
#' @param alpha Confidence level for intervals (default = 0.05).
#'
#' @return A list containing:
#' \itemize{
#'   \item{raw_results: Data frame with performance metrics}
#'   \item{interactive_table: DT::datatable interactive view}
#'   \item{final_ml_model: Trained model object}
#' }
#'
#' @details The CRAM method implements a novel approach for simultaneous model
#' training and evaluation under unknown data distributions. Key features:
#' \itemize{
#'   \item{Automated batch-wise model training}
#'   \item{Cross-validation compatible}
#'   \item{Supports both supervised and unsupervised learning}
#'   \item{Provides confidence intervals for loss estimates}
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
#' # Define the batch count (not used in this simple example)
#' nb_batch <- 5
#'
#' # Run ML learning function
#' result <- cram_ml(
#'   data = data_df,
#'   formula = Y ~ .,  # Linear regression model
#'   batch = nb_batch,
#'   loss_name = 'mse',
#'   caret_params = caret_params_lm
#' )
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
