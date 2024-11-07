# Load necessary libraries
library(grf)            # For causal forest
library(glmnet)         # For ridge regression (linear regression with penalty)
library(keras)           # For feedforward neural networks in R

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
#' @export

# Updated cram_learning function with model selection
cram_learning <- function(X, D, Y, batch, model_type = "Causal Forest", learner_type = "ridge", baseline_policy = NULL) {

  # Step 0: Set default baseline_policy if NULL
  if (is.null(baseline_policy)) {
    baseline_policy <- as.list(rep(0, nrow(X)))  # Creates a list of zeros with the same length as X
  } else {
    # Validate baseline_policy if provided
    if (!is.list(baseline_policy)) {
      stop("Error: baseline_policy must be a list.")
    }
    if (length(baseline_policy) != nrow(X)) {
      stop("Error: baseline_policy length must match the number of observations in X.")
    }
    if (!all(sapply(baseline_policy, is.numeric))) {
      stop("Error: baseline_policy must contain numeric values only.")
    }
  }

  # Step 1: Interpret `batch` argument
  if (is.numeric(batch) && length(batch) == 1) {
    # `batch` is an integer, interpret it as `nb_batch`
    nb_batch <- batch
    n <- nrow(X)
    indices <- sample(1:n)  # Randomly shuffle the indices without replacement
    group_labels <- rep(1:nb_batch, length.out = n)  # Repeat labels 1 to nb_batch, filling up to n elements
    batches <- split(indices, group_labels)  # Split indices into batches

  } else if (is.list(batch) || is.vector(batch)) {
    # `batch` is a list or vector, set `nb_batch` as its length
    batches <- batch
    nb_batch <- length(batch)
  } else {
    stop("`batch` must be either an integer or a list/vector of batch indices.")
  }

  # Step 2: Initialize a matrix to store policies
  policies <- matrix(0, nrow = n, ncol = nb_batch + 1)  # Initialize with zeros
  policies[, 1] <- unlist(baseline_policy)  # Set the first column to baseline policy

  if (model_type == "S-learner" && learner_type == "FNN") {
    s_learner_NN <- keras_model_sequential() %>%
      layer_dense(units = 64, activation = 'relu', input_shape = ncol(X) + 1) %>%
      layer_dense(units = 32, activation = 'relu') %>%
      layer_dense(units = 1)

    s_learner_NN %>% compile(
      optimizer = 'adam',
      loss = 'mse'
    )
  }

  if (model_type == "M-learner" && learner_type == "FNN") {
    m_learner_NN <- keras_model_sequential() %>%
      layer_dense(units = 64, activation = 'relu', input_shape = ncol(X)) %>%
      layer_dense(units = 32, activation = 'relu') %>%
      layer_dense(units = 1)

    m_learner_NN %>% compile(
      optimizer = 'adam',
      loss = 'mse'
    )
  }

  # Step 3: Iteratively learn policies on cumulative batches
  for (t in 1:nb_batch) {

    # Accumulate indices for batches 1 through t
    cumulative_indices <- unlist(batches[1:t])  # Combine batches 1 through t
    cumulative_X <- X[cumulative_indices, ]
    cumulative_D <- D[cumulative_indices]
    cumulative_Y <- Y[cumulative_indices]

    # Model selection based on model_type and learner_type
    if (model_type == "Causal Forest") {

      # Train causal forest on accumulated data
      causal_forest_fit <- causal_forest(cumulative_X, cumulative_Y, cumulative_D, num.trees = 2000)
      cate_estimates <- predict(causal_forest_fit, X)$predictions  # Predict CATE on full dataset X

    } else if (model_type == "S-learner") {

      if (learner_type == "ridge") {

        # Ridge Regression (S-learner)
        X_treatment <- cbind(cumulative_X, cumulative_D)
        s_learner_ridge <- cv.glmnet(as.matrix(X_treatment), cumulative_Y, alpha = 0)
        cate_estimates <- predict(s_learner_ridge, as.matrix(cbind(X, rep(1, n)))) -
          predict(s_learner_ridge, as.matrix(cbind(X, rep(0, n))))

      } else if (learner_type == "FNN") {

        # Incremental learning for Feedforward Neural Network (S-learner)
        batch_indices <- batches[[t]]
        batch_X <- X[batch_indices, ]
        batch_D <- D[batch_indices]
        batch_Y <- Y[batch_indices]

        # Combine current batch's X and D into one matrix with treatment effect
        X_treatment_batch <- cbind(batch_X, batch_D)

        # Fit the model on the current batch only (incremental learning)
        s_learner_NN %>% fit(
          as.matrix(X_treatment_batch),
          batch_Y,
          epochs = 10,         # Use fewer epochs per batch to avoid overfitting
          batch_size = 32,    # Batch size for each fit
          verbose = 0
        )

        cate_estimates <- predict(s_learner_NN, as.matrix(cbind(X, rep(1, n)))) -
          predict(s_learner_NN, as.matrix(cbind(X, rep(0, n))))
      }

    } else if (model_type == "M-learner") {

      # Propensity score estimation
      propensity_model <- glm(cumulative_D ~ ., data = as.data.frame(cumulative_X), family = "binomial")
      prop_score <- propensity_model$fitted.values

      if (learner_type == "ridge") {
        # Transformed outcome for M-learner
        Y_star <- cumulative_Y * cumulative_D / prop_score -
          cumulative_Y * (1 - cumulative_D) / (1 - prop_score)

        # Ridge Regression (M-learner)
        m_learner_ridge <- cv.glmnet(as.matrix(cumulative_X), Y_star, alpha = 0)
        cate_estimates <- predict(m_learner_ridge, as.matrix(X))

      } else if (learner_type == "FNN") {
        # Incremental learning for M-learner FNN
        batch_indices <- batches[[t]]
        batch_X <- X[batch_indices, ]
        batch_D <- D[batch_indices]
        batch_Y <- Y[batch_indices]

        # Ensure prop_score is correctly indexed for Y_star computation
        batch_prop_score <- predict(propensity_model, newdata = as.data.frame(batch_X), type = "response")

        # Transformed outcome for M-learner on current batch
        Y_star <- batch_Y * batch_D / batch_prop_score - batch_Y * (1 - batch_D) / (1 - batch_prop_score)

        # Incremental training for M-learner FNN on batch-specific data
        m_learner_NN %>% fit(
          as.matrix(batch_X),
          Y_star,
          epochs = 10,         # Use fewer epochs per batch to avoid overfitting
          batch_size = 32,     # Batch size for each fit
          verbose = 0
        )

        cate_estimates <- predict(m_learner_NN, as.matrix(X))
      }

    } else {
      stop("Unsupported model type. Choose 'Causal Forest', 'S-learner', or 'M-learner'.")
    }

    # Define the learned policy based on CATE: Assign 1 if CATE is positive, 0 otherwise
    learned_policy <- ifelse(cate_estimates > 0, 1, 0)

    # Store the learned policy in the (t+1)th column of the policies matrix
    policies[, t + 1] <- learned_policy
  }

  # Store the last fitted policy model as final_policy_learned
  final_policy_model <- if (model_type == "Causal Forest") {
    causal_forest_fit
  } else if (model_type == "S-learner" && learner_type == "ridge") {
    s_learner_ridge
  } else if (model_type == "S-learner" && learner_type == "FNN") {
    s_learner_NN
  } else if (model_type == "M-learner" && learner_type == "ridge") {
    m_learner_ridge
  } else if (model_type == "M-learner" && learner_type == "FNN") {
    m_learner_NN
  } else {
    NULL
  }

  return(list(
    final_policy_model = final_policy_model,
    policies = policies,
    batch_indices = batches
  ))
}


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
