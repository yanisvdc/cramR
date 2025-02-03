# Load necessary libraries
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R
library(doParallel)
library(foreach)

# Declare global variables to suppress devtools::check warnings
utils::globalVariables(c("X_cumul", "D_cumul", "Y_cumul", "data_cumul", "."))

#' Generalized ML Learning
#'
#' This function performs batch-wise learning for both **supervised** and **unsupervised** machine learning models.
#'
#' @param data A matrix or data frame of features. If a supervised model is used, it should also include the target variable.
#' @param formula Optional formula specifying the relationship between the target and predictors for **supervised learning** (use `NULL` for unsupervised learning).
#' @param batch Either an integer specifying the number of batches (randomly sampled) or a vector of length equal to the sample size indicating batch assignment for each observation.
#' @param parallelize_batch Logical. Whether to parallelize batch processing. Defaults to `FALSE`.
#'  - If `TRUE`, batch models are trained in parallel.
#'  - If `FALSE`, training is performed sequentially using `data.table` for efficiency.
#' @param loss_name The name of the loss function to be used (e.g., `"mse"`, `"logloss"`).
#' @param caret_params A **list** of parameters to pass to the `caret::train()` function.
#'   - Required: `method` (e.g., `"glm"`, `"rf"`, `"kmeans"`).
#'   - If `method = "kmeans"`, the function will automatically return **cluster assignments**.
#' @param custom_fit A **custom function** for training user-defined models. Defaults to `NULL`.
#' @param custom_predict A **custom function** for making predictions from user-defined models. Defaults to `NULL`.
#' @param custom_loss Optional **custom function** for computing the loss of a trained model on the data. Should return a **vector** containing per-instance losses.
#' @param n_cores Number of CPU cores to use for parallel processing (`parallelize_batch = TRUE`). Defaults to `detectCores() - 1`.
#'
#' @return A **list** containing:
#'   \item{final_ml_model}{The final trained ML model.}
#'   \item{losses}{A matrix of losses where each column represents a batch's trained model. The first column contains zeros (baseline model).}
#'   \item{batch_indices}{The indices of observations in each batch.}
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
#' result_lm <- ml_learning(
#'   data = data_df,
#'   formula = Y ~ .,  # Linear regression model
#'   batch = nb_batch,
#'   loss_name = 'mse',
#'   caret_params = caret_params_lm
#' )
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
      data_subset <- data[cumulative_indices, ]

      ## FIT and PREDICT
      if (!(is.null(caret_params))) {
        # Caret model
        trained_model <- fit_model_ml(data_subset, formula, caret_params)
        ml_preds <- model_predict_ml(trained_model, data, formula, caret_params)
      } else {
        # Custom model
        trained_model <- custom_fit(data_subset)
        ml_preds <- custom_predict(trained_model, data)
      }

      ## LOSS CALCULATION
      if (!(is.null(loss_name))) {
        loss_vec <- compute_loss(ml_preds, data, formula, loss_name)
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
      losses = loss_matrix,
      batch_indices = batches
    ))


    # SEQUENTIAL CRAM PROCEDURE -----------------------------------------------

  } else {

    # Store cumulative data for each step of the cram procedure
    cumulative_data_dt <- create_cumulative_data_ml(
      data = data,
      batches = batches,
      nb_batch = nb_batch
    )

    # Use data.table structure to handle fit and predict for each step
    results_dt <- cumulative_data_dt[, {
      # Extract cumulative X, D, Y for the current cumulative batches (1:t)
      data_subset <- as.matrix(data_cumul[[1]])

      ## FIT and PREDICT
      if (!(is.null(caret_params))) {
        # Caret model
        trained_model <- fit_model_ml(data_subset, formula, caret_params)
        ml_preds <- model_predict_ml(trained_model, data, formula, caret_params)

      } else {
        # Custom model
        trained_model <- custom_fit(data_subset)
        ml_preds <- custom_predict(trained_model, data)
      }

      ## LOSS CALCULATION
      if (!(is.null(loss_name))) {
        loss_vec <- compute_loss(ml_preds, data, formula, loss_name)
      } else {
        loss_vec <- custom_loss(ml_preds, data)
      }

      ## FINAL MODEL
      final_model <- if (t == nb_batch) trained_model else NULL

      .(loss = list(loss_vec), final_model=list(final_model))
    }, by = t]

    # Extract loss list
    loss_list <- results_dt$loss

    # Convert the list of losses into a matrix
    loss_matrix <- do.call(cbind, lapply(loss_list, as.numeric))

    # Add zeros as the first column
    zero_column <- matrix(0, nrow = nrow(loss_matrix), ncol = 1)
    loss_matrix <- cbind(zero_column, loss_matrix)

    final_ml_model <- results_dt$final_model[[nb_batch]]

    return(list(
      final_ml_model = final_ml_model,
      losses = loss_matrix,
      batch_indices = batches
    ))
  }
}
