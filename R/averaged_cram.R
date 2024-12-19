#' Averaged CRAM with Permutations
#'
#' This function implements Averaged CRAM by randomly permuting batches and averaging performance results.
#' @param X A matrix or data frame of covariates.
#' @param D A binary vector of treatment indicators (0 or 1).
#' @param Y A vector of outcomes.
#' @param batch Number of batches for CRAM or batch indices.
#' @param model_type Type of model for CRAM ("causal_forest", "s_learner", "m_learner").
#' @param learner_type Type of learner ("ridge", "fnn", or NULL).
#' @param alpha Confidence level for performance estimation.
#' @param baseline_policy Baseline policy vector.
#' @param parallelize_batch Boolean to parallelize batch processing.
#' @param model_params Additional parameters for the model.
#' @param num_permutations Number of random permutations of batches.
#' @return A list with averaged performance and variance estimates.
#' @examples
#' avg_cram_results <- averaged_cram(X, D, Y, batch = 20, model_type = "m_learner", learner_type = "ridge")
averaged_cram <- function(X, D, Y, batch, model_type, learner_type = NULL,
                          alpha = 0.05, baseline_policy = NULL,
                          parallelize_batch = FALSE, model_params = NULL,
                          custom_fit = NULL, custom_predict = NULL,
                          num_permutations = 10) {
  n <- nrow(X)
  T <- if (is.numeric(batch)) batch else length(batch)

  if (is.null(baseline_policy)) {
    baseline_policy <- as.list(rep(0, n)) # Default to zero policy
  }

  # Generate L random permutations of indices for T-1 training batches
  batch_indices <- test_batch(batch, n)$batches
  train_batches <- batch_indices[1:(T - 1)] # Leave the last batch for evaluation

  # Combine batches into one large vector of indices
  all_indices <- unlist(train_batches)

  # Generate L random permutations
  permutations <- replicate(num_permutations, sample(all_indices), simplify = FALSE)

  # Store results for each permutation
  results <- vector("list", length = num_permutations)

  for (l in seq_len(num_permutations)) {
    permuted_batches <- split(permutations[[l]], rep(1:(T - 1), each = length(permutations[[l]]) / (T - 1)))
    permuted_batches[[T]] <- batch_indices[[T]] # Add the holdout batch

    # Run CRAM experiment with permuted batches
    results[[l]] <- cram_experiment(
      X = X, D = D, Y = Y, batch = permuted_batches,
      model_type = model_type, learner_type = learner_type,
      alpha = alpha, baseline_policy = baseline_policy,
      parallelize_batch = parallelize_batch, model_params = model_params
    )
  }

  # Aggregate results
  policy_values <- sapply(results, function(res) res$policy_value_estimate)
  avg_policy_value <- mean(policy_values)
  var_policy_value <- var(policy_values)

  return(list(
    avg_policy_value = avg_policy_value,
    var_policy_value = var_policy_value,
    all_results = results
  ))
}

# Example usage of Averaged CRAM
set.seed(123)

# Generate synthetic data
n <- 1000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

# Parameters
batch <- 20
model_type <- "m_learner"
learner_type <- "ridge"
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X)))
parallelize_batch <- TRUE
model_params <- NULL
num_permutations <- 10

# Run Averaged CRAM
avg_cram_results <- averaged_cram(
  X = X, D = D, Y = Y, batch = batch,
  model_type = model_type, learner_type = learner_type,
  alpha = alpha, baseline_policy = baseline_policy,
  parallelize_batch = parallelize_batch, model_params = model_params,
  num_permutations = num_permutations
)

# Print Results
print(avg_cram_results$avg_policy_value)
print(avg_cram_results$var_policy_value)
