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

  # Step 0: Test baseline_policy
  baseline_policy <- test_baseline_policy(baseline_policy, n)

  # Step 1: Interpret `batch` argument
  batch_results <- test_batch(batch, n)
  batches <- batch_results$batches
  nb_batch <- batch_results$nb_batch

  if (!(is.null(model_type))) {
    # Step 2: Retrieve model and validate user-specified parameters
    if (!is.null(learner_type) && learner_type == "fnn") {
      model_params <- validate_params_fnn(model_type, learner_type, model_params, X)
      model <- set_model(model_type, learner_type, model_params)
    } else {
      model <- set_model(model_type, learner_type, model_params)
      model_params <- validate_params(model, model_type, learner_type, model_params)
    }
  } else {
    if (is.null(custom_fit) || is.null(custom_predict)) {
      stop("As model_type is NULL (custom mode), custom_fit and custom_predict must be specified")
    }
  }


  if (parallelize_batch) {

    # Parallel execution using foreach and doParallel
    cl <- makeCluster(n_cores)  # Use available cores minus one
    registerDoParallel(cl)

    if (!is.null(learner_type) && learner_type == "fnn") {
      clusterExport(cl, varlist = c("X", "D", "set_model",
                                    "model_type", "learner_type", "model_params",
                                    "fit_model", "model_predict"), envir = environment())
    } else {
      if (!(is.null(model_type))) {
        # Export custom functions and objects to the worker nodes
        clusterExport(cl, varlist = c("X", "D",
                                      "model_type", "learner_type", "model_params",
                                      "fit_model", "model_predict"), envir = environment())
      } else {
        # Custom model
        clusterExport(cl, varlist = c("X", "D", "custom_fit",
                                      "custom_predict", "fit_model", "model_predict"), envir = environment())
      }
    }

    # clusterEvalQ(cl, {
    #   library(grf)
    #   library(glmnet)
    #   library(keras)
    #   library(data.table)
    # })

    # Perform parallel training
    results <- foreach(t = 1:nb_batch, .packages = c("grf", "data.table",
                                                     "glmnet", "keras")) %dopar% {

      cumulative_indices <- unlist(batches[1:t])
      X_subset <- as.matrix(X[cumulative_indices, ])
      D_subset <- as.numeric(D[cumulative_indices])
      Y_subset <- as.numeric(Y[cumulative_indices])

      # I need to set the keras model in each worker because keras cannot be transmitted
      # to the workers
      if (!(is.null(model_type))) {
        if (!is.null(learner_type) && learner_type == "fnn") {
          model <- set_model(model_type, learner_type, model_params)
        }
      }

      # Train model with validated parameters
      if (!(is.null(model_type))) {
        trained_model <- fit_model(model, X_subset, Y_subset, D_subset, model_type, learner_type, model_params)
        cate_estimates <- model_predict(trained_model, X, D, model_type, learner_type, model_params)
      } else {
        trained_model <- custom_fit(X_subset, Y_subset, D_subset)
        cate_estimates <- custom_predict(trained_model, X, D)
      }
      cate_estimates <- as.numeric(cate_estimates)
      learned_policy <- ifelse(cate_estimates > 0, 1, 0)

      if (!is.null(learner_type) && learner_type == "fnn") {
        # Serialize the final model at the last iteration
        final_model <- if (t == nb_batch) serialize_model(trained_model) else NULL
      } else {
        # Store the final model only at the last iteration
        final_model <- if (t == nb_batch) trained_model else NULL
      }

      # Return the policy matrix - foreach preserves the sequential order when rendering the output
      list(learned_policy = learned_policy, final_model = final_model)
    }

    stopCluster(cl)
    foreach::registerDoSEQ()
    # closeAllConnections()

    # Combine results into a data.table
    results_dt <- results

    # Combine the learned policies into a matrix
    policy_matrix <- do.call(cbind, lapply(results_dt, function(x) x$learned_policy))

    # Add a baseline policy as the first column (optional)
    policy_matrix <- cbind(as.numeric(baseline_policy), policy_matrix)

    if (!is.null(learner_type) && learner_type == "fnn") {
      serialized_model <- results_dt[[nb_batch]]$final_model
      final_policy_model <- unserialize_model(serialized_model)
    } else {
      # Extract the final model from the last iteration
      final_policy_model <- results_dt[[nb_batch]]$final_model
    }

    return(list(
      final_policy_model = final_policy_model,
      policies = policy_matrix,
      batch_indices = batches
    ))

  } else {

    # Step 3: Create a data.table for cumulative batches
    # Initialize an empty list to store cumulative data for each batch
    cumulative_data_list <- lapply(1:nb_batch, function(t) {
      # Combine indices for batches 1 through t
      cumulative_indices <- unlist(batches[1:t])

      # Subset X, D, Y using cumulative indices
      list(
        t = t,  # Add t as the index
        # cumulative_index = list(cumulative_indices),  # Store cumulative indices as a list in one row
        X_cumul = list(X[cumulative_indices, ]),
        D_cumul = list(D[cumulative_indices]),
        Y_cumul = list(Y[cumulative_indices])
      )
    })

    # Convert the list to a data.table
    cumulative_data_dt <- rbindlist(cumulative_data_list)

    results_dt <- cumulative_data_dt[, {
      # Extract cumulative X, D, Y for the current batch (t)
      X_subset <- as.matrix(X_cumul[[1]])
      D_subset <- as.numeric(D_cumul[[1]])
      Y_subset <- as.numeric(Y_cumul[[1]])

      # Train model with validated parameters
      if (!(is.null(model_type))) {
        trained_model <- fit_model(model, X_subset, Y_subset, D_subset, model_type, learner_type, model_params)
        cate_estimates <- model_predict(trained_model, X, D, model_type, learner_type, model_params)
      } else {
        trained_model <- custom_fit(X_subset, Y_subset, D_subset)
        cate_estimates <- custom_predict(trained_model, X, D)
      }

      learned_policy <- ifelse(cate_estimates > 0, 1, 0)

      final_model <- if (t == nb_batch) trained_model else NULL

      .(learned_policy = list(learned_policy), final_model=list(final_model))
    }, by = t]

  # Extract learned_policy list
  learned_policy_list <- results_dt$learned_policy

  # Convert the list of learned policies into a matrix
  learned_policy_matrix <- do.call(cbind, lapply(learned_policy_list, as.numeric))

  # Combine baseline_policy as the first column
  policy_matrix <- cbind(as.numeric(baseline_policy), learned_policy_matrix)

  final_policy_model <- results_dt$final_model[[nb_batch]]

  return(list(
        final_policy_model = final_policy_model,
        policies = policy_matrix,
        batch_indices = batches
      ))
  }
}
