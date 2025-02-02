# Load necessary libraries
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R
library(doParallel)
library(foreach)

# Declare global variables to suppress devtools::check warnings
utils::globalVariables(c("X_cumul", "D_cumul", "Y_cumul", "."))

#' Generalized ML Learning
#'
#' This function performs batch-wise learning for any ML model.
#'
#' @param data A matrix or data frame of data, including or not a target variable (handles both supervised and unsupervised ML models).
#' @param formula Optional formula relating the target to the predictors for supervised learning (NULL for unsupervised)
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a vector of length equal to the sample size providing the batch assignment (index) for each individual in the sample.
#' @param parallelize_batch Logical. Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches. They are learned in parallel when parallelize_batch is TRUE vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training). Defaults to \code{FALSE}.
#' @param loss_name The name of the loss the user wants to use.
#' @param caret_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @param custom_fit A custom, user-defined, function that outputs a fitted model given training data (allows flexibility). Defaults to \code{NULL}.
#' @param custom_predict A custom, user-defined, function for making predictions given a fitted model and test data (allow flexibility). Defaults to \code{NULL}.
#' @param custom_loss Optional a custom, user-defined, function for calculating the loss of a fitted ML model on some data; data should be a matrix or data frame and the function should return a vector containing the loss for each individual.
#' @param n_cores Number of cores to use for parallelization when parallelize_batch is set to TRUE. Defaults to detectCores() - 1.
#' @return A list containing:
#'   \item{final_ml_model}{The final fitted ML model, depending on \code{model_type} and \code{model_params}.}
#'   \item{losses}{A matrix of losses, where each column represents a batch's learned ML model and the first column is all zeroes (baseline ML model)}
#'   \item{batch_indices}{The indices for each batch, either as generated (if \code{batch} is an integer) or as provided by the user.}
#' @examples
#' # Example usage
#' X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)
#' D_data <- sample(c(0, 1), 100, replace = TRUE)
#' Y_data <- rnorm(100)
#' nb_batch <- 5
#'
#' # Perform CRAM learning
#' result <- cram_learning(X = X_data, D = D_data, Y = Y_data, batch = nb_batch)
#'
#' # Access the learned policies and final model
#' policies_matrix <- result$policies
#' final_model <- result$final_policy_model
#' batch_indices <- result$batch_indices
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{keras_model_sequential}}
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
#' @export
ml_learning <- function(data, formula=NULL, batch,
                        parallelize_batch = FALSE, loss_name = "mse",
                        caret_params = NULL, custom_fit = NULL,
                        custom_predict = NULL, custom_loss = NULL,
                        n_cores = detectCores() - 1) {

  n <- nrow(data)

  # Process `batch` argument
  batch_results <- test_batch(batch, n)
  batches <- batch_results$batches
  nb_batch <- batch_results$nb_batch

  # PARALLEL CRAM PROCEDURE -------------------------------------------------

  if (parallelize_batch) {

    # Parallel execution using foreach and doParallel
    cl <- makeCluster(n_cores)  # Use number of cores specified by the user
    registerDoParallel(cl)

    # Variables to export to the different workers
    varlist <- c("data", "formula", "fit_model_ml", "model_predict_ml")

    # Export the variables to the cluster
    clusterExport(cl, varlist = varlist, envir = environment())

    # Define the list of required packages
    required_packages <- c("caret", "data.table")

    results <- foreach(t = 1:nb_batch, .packages = required_packages) %dopar% {

      cumulative_indices <- unlist(batches[1:t])
      data_subset <- as.matrix(data[cumulative_indices, ])

      ## FIT and PREDICT
      if (!(is.null(caret_params))) {
        # Caret model
        trained_model <- fit_model_ml(data_subset, formula, caret_params)
        ml_preds <- model_predict_ml(trained_model, data, formula)
      } else {
        # Custom model
        trained_model <- custom_fit(data_subset)
        ml_preds <- custom_predict(trained_model, data)
      }

      ## LOSS CALCULATION
      if (!(is.null(custom_loss))) {
        loss_vec <- compute_loss(ml_preds, data)
      } else {
        loss_vec <- custom_loss(ml_preds, data)
      }

      ## FINAL MODEL
      final_model <- if (t == nb_batch) trained_model else NULL


      # Return the policy matrix - foreach preserves the sequential order when rendering the output
      list(loss = loss_vec, final_model = final_model)
    }

    stopCluster(cl)
    foreach::registerDoSEQ()

    # Combine the learned policies into a matrix
    loss_matrix <- do.call(cbind, lapply(results, function(x) x$loss))

    # Add a baseline loss (all zeros) as the first column
    zero_column <- matrix(0, nrow = nrow(loss_matrix), ncol = 1)
    loss_matrix <- cbind(zero_column, loss_matrix)

    # Retrieve final model
    final_ml_model <- results[[nb_batch]]$final_model


    return(list(
      final_ml_model = final_ml_model,
      policies = policy_matrix,
      batch_indices = batches
    ))


    # SEQUENTIAL CRAM PROCEDURE -----------------------------------------------

  } else {

    # Store cumulative data for each step of the cram procedure
    cumulative_data_dt <- create_cumulative_data(
      X = X,
      D = D,
      Y = Y,
      batches = batches,
      nb_batch = nb_batch
    )

    # Use data.table structure to handle fit and predict for each step
    results_dt <- cumulative_data_dt[, {
      # Extract cumulative X, D, Y for the current cumulative batches (1:t)
      X_subset <- as.matrix(X_cumul[[1]])
      D_subset <- as.numeric(D_cumul[[1]])
      Y_subset <- as.numeric(Y_cumul[[1]])

      ## FIT and PREDICT
      if (!(is.null(model_type))) {
        # Package model
        trained_model <- fit_model(model, X_subset, Y_subset, D_subset, model_type, learner_type, model_params)
        learned_policy <- model_predict(trained_model, X, D, model_type, learner_type, model_params)
      } else {
        # Custom model
        trained_model <- custom_fit(X_subset, Y_subset, D_subset)
        learned_policy <- custom_predict(trained_model, X, D)
      }

      ## FINAL MODEL
      final_model <- if (t == nb_batch) trained_model else NULL

      .(learned_policy = list(learned_policy), final_model=list(final_model))
    }, by = t]

    # Extract learned_policy list
    learned_policy_list <- results_dt$learned_policy

    # Convert the list of learned policies into a matrix
    learned_policy_matrix <- do.call(cbind, lapply(learned_policy_list, as.numeric))

    # Add baseline_policy as the first column
    policy_matrix <- cbind(as.numeric(baseline_policy), learned_policy_matrix)

    final_policy_model <- results_dt$final_model[[nb_batch]]

    return(list(
      final_policy_model = final_policy_model,
      policies = policy_matrix,
      batch_indices = batches
    ))
  }
}
