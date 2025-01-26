# Load necessary libraries
library(grf)            # For causal forest
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R
library(doParallel)
library(foreach)

# Declare global variables to suppress devtools::check warnings
utils::globalVariables(c("X_cumul", "D_cumul", "Y_cumul", "."))

#' CRAM Learning
#'
#' This function performs the learning part of the cram method.
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for untreated).
#' @param Y A vector of outcome values for each sample.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a vector of length equal to the sample size providing the batch assignment (index) for each individual in the sample.
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param baseline_policy A list providing the baseline policy (binary 0 or 1) for each sample. If \code{NULL}, the baseline policy defaults to a list of zeros with the same length as the number of rows in \code{X}.
#' @param parallelize_batch Logical. Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches. They are learned in parallel when parallelize_batch is TRUE vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training). Defaults to \code{FALSE}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @param custom_fit A custom, user-defined, function that outputs a fitted model given training data (allows flexibility). Defaults to \code{NULL}.
#' @param custom_predict A custom, user-defined, function for making predictions given a fitted model and test data (allow flexibility). Defaults to \code{NULL}.
#' @param n_cores Number of cores to use for parallelization when parallelize_batch is set to TRUE. Defaults to detectCores() - 1.
#' @return A list containing:
#'   \item{final_policy_model}{The final fitted policy model, depending on \code{model_type} and \code{learner_type}.}
#'   \item{policies}{A matrix of learned policies, where each column represents a batch's learned policy and the first column is the baseline policy.}
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
cram_learning <- function(X, D, Y, batch, model_type = "causal_forest",
                          learner_type = "ridge", baseline_policy = NULL,
                          parallelize_batch = FALSE, model_params = NULL,
                          custom_fit = NULL, custom_predict = NULL, n_cores = detectCores() - 1) {

  n <- nrow(X)

  # Check for mismatched lengths
  check_lengths(D, Y, n = n)

  # Process baseline_policy
  baseline_policy <- test_baseline_policy(baseline_policy, n)

  # Process `batch` argument
  batch_results <- test_batch(batch, n)
  batches <- batch_results$batches
  nb_batch <- batch_results$nb_batch

  # Process model and model_params
  model_info <- retrieve_and_validate_model(
    model_type = model_type,
    learner_type = learner_type,
    model_params = model_params,
    X = X,
    custom_fit = custom_fit,
    custom_predict = custom_predict
  )

  # Extract model and model_params
  model <- model_info$model
  model_params <- model_info$model_params

  # PARALLEL CRAM PROCEDURE -------------------------------------------------

  if (parallelize_batch) {

    # Parallel execution using foreach and doParallel
    cl <- makeCluster(n_cores)  # Use number of cores specified by the user
    registerDoParallel(cl)


    # Export variables to cluster
    export_cluster_variables(
      cl = cl,
      learner_type = learner_type,
      model_type = model_type,
      model_params = model_params,
      custom_fit = custom_fit,
      custom_predict = custom_predict
    )

    # Define the list of required packages
    required_packages <- c("grf", "data.table", "glmnet", "keras")

    results <- foreach(t = 1:nb_batch, .packages = required_packages) %dopar% {

      cumulative_indices <- unlist(batches[1:t])
      X_subset <- as.matrix(X[cumulative_indices, ])
      D_subset <- as.numeric(D[cumulative_indices])
      Y_subset <- as.numeric(Y[cumulative_indices])


      ## SET KERAS MODEL IN EACH WORKER
      # I need to set the keras model in each worker
      # because keras structure cannot be exported to the workers
      if (!(is.null(model_type))) {
        if (!is.null(learner_type) && learner_type == "fnn") {
          model <- set_model(model_type, learner_type, model_params)
        }
      }

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
      if (!is.null(learner_type) && learner_type == "fnn") {
        # KERAS: serialize the final model at the last iteration
        final_model <- if (t == nb_batch) serialize_model(trained_model) else NULL
      } else {
        # Any other model
        final_model <- if (t == nb_batch) trained_model else NULL
      }

      # Return the policy matrix - foreach preserves the sequential order when rendering the output
      list(learned_policy = learned_policy, final_model = final_model)
    }

    stopCluster(cl)
    foreach::registerDoSEQ()

    # Combine the learned policies into a matrix
    policy_matrix <- do.call(cbind, lapply(results, function(x) x$learned_policy))

    # Add a baseline policy as the first column (optional)
    policy_matrix <- cbind(as.numeric(baseline_policy), policy_matrix)

    if (!is.null(learner_type) && learner_type == "fnn") {
      # KERAS: unserialize the final policy model
      serialized_model <- results[[nb_batch]]$final_model
      final_policy_model <- unserialize_model(serialized_model)
    } else {
      # Any other model
      final_policy_model <- results[[nb_batch]]$final_model
    }

    return(list(
      final_policy_model = final_policy_model,
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
