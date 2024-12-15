# Load necessary libraries
library(grf)            # For causal forest
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R
library(doParallel)
library(foreach)

#' CRAM Learning with Model Selection
#'
#' This function performs policy learning using cumulative batches with a choice of model types and learner types. Supported models include Causal Forest, S-learner, and M-learner with options for Ridge Regression and Feedforward Neural Network (FNN) learners.
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param D A vector of binary treatment indicators (1 for treated, 0 for untreated).
#' @param Y A vector of outcome values for each sample.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a list/vector providing specific batch indices.
#' @param model_type The model type for policy learning. Options include \code{"Causal Forest"}, \code{"S-learner"}, and \code{"M-learner"}. Default is \code{"Causal Forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"FNN"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param baseline_policy A list providing the baseline policy (binary 0 or 1) for each sample. If \code{NULL}, the baseline policy defaults to a list of zeros with the same length as the number of samples in \code{X}.
#' @return A list containing:
#'   \item{final_policy_model}{The final fitted policy model, depending on \code{model_type} and \code{learner_type}.}
#'   \item{policies}{A matrix of learned policies, where each column represents a batch's learned policy and the first column is the baseline policy.}
#'   \item{batch_indices}{The indices for each batch, either as generated (if \code{batch} is an integer) or as provided by the user.}
#' @examples
#' # Example usage
#' X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)  # 100 samples, 5 features
#' D_data <- sample(c(0, 1), 100, replace = TRUE)          # Random binary treatment assignment
#' Y_data <- rnorm(100)                                    # Random outcome variable
#' nb_batch <- 3                                           # Number of batches
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
#' @export
cram_learning <- function(X, D, Y, batch, model_type = "causal_forest",
                          learner_type = "ridge", baseline_policy = NULL,
                          parallelize_batch = FALSE, model_params = NULL) {

  n <- nrow(X)

  # Step 0: Test baseline_policy
  baseline_policy <- test_baseline_policy(baseline_policy, n)

  # Step 1: Interpret `batch` argument
  batch_results <- test_batch(batch, n)
  batches <- batch_results$batches
  nb_batch <- batch_results$nb_batch

  # Step 2: Retrieve model and validate user-specified parameters
  if (learner_type == "fnn") {
    model_params <- validate_params_fnn(model_type, learner_type, model_params)
    model <- set_model(model_type, learner_type, model_params)
  } else {
    model <- set_model(model_type, learner_type, model_params)
    model_params <- validate_params(model, model_type, learner_type, model_params)
  }

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

  if (parallelize_batch) {
    # Parallel execution using foreach and doParallel
    cl <- makeCluster(detectCores() - 1)  # Use available cores minus one
    registerDoParallel(cl)

    # Perform parallel training
    results <- foreach(t = 1:nb_batch, .combine = rbind, .packages = c("grf", "data.table")) %dopar% {
      batch <- cumulative_data_dt[t]
      X_subset <- batch$X_cumul[[1]]
      D_subset <- batch$D_cumul[[1]]
      Y_subset <- batch$Y_cumul[[1]]

      # Train model with validated parameters
      trained_model <- fit_model(model, X_subset, Y_subset, D_subset, model_type, learner_type, model_params)
      # trained_model <- do.call(model, c(list(X = X_subset, Y = Y_subset, W = D_subset), model_params))
      cate_estimates <- predict(trained_model, X_subset)$predictions
      learned_policy <- ifelse(cate_estimates > 0, 1, 0)

      list(t = t, model = trained_model, cate_estimates = cate_estimates, learned_policy = learned_policy)
    }

    stopCluster(cl)

    # Combine results into a data.table
    results_dt <- rbindlist(results)

  } else {

  results_dt <- cumulative_data_dt[, {
    # Extract cumulative X, D, Y for the current batch (t)
    X_subset <- as.matrix(X_cumul[[1]])
    D_subset <- as.numeric(D_cumul[[1]])
    Y_subset <- as.numeric(Y_cumul[[1]])

    # Train model with validated parameters
    trained_model <- fit_model(model, X_subset, Y_subset, D_subset, model_type, learner_type, model_params)
    cate_estimates <- model_predict(trained_model, X, D, model_type, learner_type, model_params)
    learned_policy <- ifelse(cate_estimates > 0, 1, 0)

    # Return trained model only for the last batch
    if (t == max(cumulative_data_dt$t)) {
      assign("final_policy_model", trained_model, envir = .GlobalEnv)
    }

    # (model = list(trained_model), cate_estimates = list(cate_estimates),
    .(learned_policy = list(learned_policy))
  }, by = t]

  # Extract learned_policy list
  learned_policy_list <- results_dt$learned_policy

  # Convert the list of learned policies into a matrix
  learned_policy_matrix <- do.call(cbind, lapply(learned_policy_list, as.numeric))

  # Combine baseline_policy as the first column
  policy_matrix <- cbind(as.numeric(baseline_policy), learned_policy_matrix)

  return(list(
        final_policy_model = final_policy_model,
        policies = policy_matrix,
        batch_indices = batches
      ))

  }

}

#   # Step 3: Iteratively learn policies on cumulative batches
#   for (t in 1:nb_batch) {
#
#     # Accumulate indices for batches 1 through t
#     cumulative_indices <- unlist(batches[1:t])  # Combine batches 1 through t
#     cumulative_X <- X[cumulative_indices, ]
#     cumulative_D <- D[cumulative_indices]
#     cumulative_Y <- Y[cumulative_indices]
#
#     # Model selection based on model_type and learner_type
#     if (model_type == "Causal Forest") {
#
#       # Train causal forest on accumulated data
#       causal_forest_fit <- causal_forest(cumulative_X, cumulative_Y, cumulative_D, num.trees = 100)
#       cate_estimates <- predict(causal_forest_fit, X)$predictions  # Predict CATE on full dataset X
#
#     } else if (model_type == "S-learner") {
#
#       if (learner_type == "ridge") {
#
#         # Ridge Regression (S-learner)
#         X_treatment <- cbind(cumulative_X, cumulative_D)
#         s_learner_ridge <- cv.glmnet(as.matrix(X_treatment), cumulative_Y, alpha = 0)
#         cate_estimates <- predict(s_learner_ridge, as.matrix(cbind(X, rep(1, n)))) -
#           predict(s_learner_ridge, as.matrix(cbind(X, rep(0, n))))
#
#       } else if (learner_type == "FNN") {
#
#         # Incremental learning for Feedforward Neural Network (S-learner)
#         batch_indices <- batches[[t]]
#         batch_X <- X[batch_indices, ]
#         batch_D <- D[batch_indices]
#         batch_Y <- Y[batch_indices]
#
#         # Combine current batch's X and D into one matrix with treatment effect
#         X_treatment_batch <- cbind(batch_X, batch_D)
#
#         # Fit the model on the current batch only (incremental learning)
#         s_learner_NN %>% fit(
#           as.matrix(X_treatment_batch),
#           batch_Y,
#           epochs = 10,         # Use fewer epochs per batch to avoid overfitting
#           batch_size = 32,    # Batch size for each fit
#           verbose = 0
#         )
#
#         cate_estimates <- predict(s_learner_NN, as.matrix(cbind(X, rep(1, n)))) -
#           predict(s_learner_NN, as.matrix(cbind(X, rep(0, n))))
#       }
#
#     } else if (model_type == "M-learner") {
#
#       # Propensity score estimation
#       propensity_model <- glm(cumulative_D ~ ., data = as.data.frame(cumulative_X), family = "binomial")
#       prop_score <- propensity_model$fitted.values
#
#       if (learner_type == "ridge") {
#         # Transformed outcome for M-learner
#         Y_star <- cumulative_Y * cumulative_D / prop_score -
#           cumulative_Y * (1 - cumulative_D) / (1 - prop_score)
#
#         # Ridge Regression (M-learner)
#         m_learner_ridge <- cv.glmnet(as.matrix(cumulative_X), Y_star, alpha = 0)
#         cate_estimates <- predict(m_learner_ridge, as.matrix(X))
#
#       } else if (learner_type == "FNN") {
#         # Incremental learning for M-learner FNN
#         batch_indices <- batches[[t]]
#         batch_X <- X[batch_indices, ]
#         batch_D <- D[batch_indices]
#         batch_Y <- Y[batch_indices]
#
#         # Ensure prop_score is correctly indexed for Y_star computation
#         batch_prop_score <- predict(propensity_model, newdata = as.data.frame(batch_X), type = "response")
#
#         # Transformed outcome for M-learner on current batch
#         Y_star <- batch_Y * batch_D / batch_prop_score - batch_Y * (1 - batch_D) / (1 - batch_prop_score)
#
#         # Incremental training for M-learner FNN on batch-specific data
#         m_learner_NN %>% fit(
#           as.matrix(batch_X),
#           Y_star,
#           epochs = 10,         # Use fewer epochs per batch to avoid overfitting
#           batch_size = 32,     # Batch size for each fit
#           verbose = 0
#         )
#
#         cate_estimates <- predict(m_learner_NN, as.matrix(X))
#       }
#
#     } else {
#       stop("Unsupported model type. Choose 'Causal Forest', 'S-learner', or 'M-learner'.")
#     }
#
#     # Define the learned policy based on CATE: Assign 1 if CATE is positive, 0 otherwise
#     learned_policy <- ifelse(cate_estimates > 0, 1, 0)
#
#     # Store the learned policy in the (t+1)th column of the policies matrix
#     policies[, t + 1] <- learned_policy
#   }
#
#   # Store the last fitted policy model as final_policy_learned
#   final_policy_model <- if (model_type == "Causal Forest") {
#     causal_forest_fit
#   } else if (model_type == "S-learner" && learner_type == "ridge") {
#     s_learner_ridge
#   } else if (model_type == "S-learner" && learner_type == "FNN") {
#     s_learner_NN
#   } else if (model_type == "M-learner" && learner_type == "ridge") {
#     m_learner_ridge
#   } else if (model_type == "M-learner" && learner_type == "FNN") {
#     m_learner_NN
#   } else {
#     NULL
#   }
#
#   return(list(
#     final_policy_model = final_policy_model,
#     policies = policies,
#     batch_indices = batches
#   ))
# }


# Example usage:
# Assuming X, D, Y are already defined and nb_batch is set
# result <- cram_learning(X = X_data, D = D_data, Y = Y_data, nb_batch = 3)
# This will return a list `result` where:
# - result$policies is the policies matrix,
# - result$batch_indices contains the batch indices in the format of split(1:100, rep(1:nb_batch, each = 10)).

# Example usage:
# Assuming X, D, Y are already defined and nb_batch is set
# policies_matrix <- cram_learning(X = X_data, D = D_data, Y = Y_data, nb_batch = 3)
# This will return a matrix `policies_matrix` where:
# - policies_matrix[, 1] is the baseline policy (all 0s),
# - policies_matrix[, 2] to policies_matrix[, nb_batch + 1] are the learned policies on cumulative batches.
