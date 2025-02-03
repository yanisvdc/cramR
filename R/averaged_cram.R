#' Averaged CRAM with Permutations
#'
#' This function implements Averaged CRAM by randomly permuting batches
#' (except the last batch which is kept the same) and averaging performance results.
#' @param X A matrix or data frame of covariates.
#' @param D A binary vector of treatment indicators (0 or 1).
#' @param Y A vector of outcomes.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a vector of length equal to the sample size providing the batch assignment (index) for each individual in the sample.
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param alpha Significance level for confidence intervals. Default is 0.05 (95\% confidence).
#' @param baseline_policy A list providing the baseline policy (binary 0 or 1) for each sample. If \code{NULL}, defaults to a list of zeros with the same length as the number of rows in \code{X}.
#' @param parallelize_batch Logical. Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches. They are learned in parallel when parallelize_batch is TRUE vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training). Defaults to \code{FALSE}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#' @param custom_fit A custom, user-defined, function that outputs a fitted model given training data (allows flexibility). Defaults to \code{NULL}.
#' @param custom_predict A custom, user-defined, function for making predictions given a fitted model and test data (allow flexibility). Defaults to \code{NULL}.
#' @param num_permutations Number of random permutations of batches.
#' @return A list with averaged performance and variance estimates.
#' @examples
#' X <- matrix(rnorm(1000), nrow = 100, ncol = 10)
#' D <- sample(0:1, 100, replace = TRUE)
#' Y <- rnorm(100)
#' avg_cram_results <- averaged_cram(X, D, Y,
#'                                   batch = 20,
#'                                   model_type = "m_learner",
#'                                   learner_type = "ridge",
#'                                   num_permutations = 3)
#' @export
averaged_cram <- function(X, D, Y, batch, model_type, learner_type = NULL,
                          alpha = 0.05, baseline_policy = NULL,
                          parallelize_batch = FALSE, model_params = NULL,
                          custom_fit = NULL, custom_predict = NULL,
                          num_permutations = 10) {
  n <- nrow(X)

  # Step 0: Test baseline_policy
  baseline_policy <- test_baseline_policy(baseline_policy, n)

  # Step 1: Interpret `batch` argument
  batch_results <- test_batch(batch, n)
  batches <- batch_results$batches
  nb_batch <- batch_results$nb_batch
  batch_size <- length(batch_results$batches[1])

  # Generate L random permutations of indices for T-1 training batches
  train_batches <- batches[1:(nb_batch - 1)] # Leave the last batch for evaluation

  # Combine batches into one large vector of indices
  all_indices <- unlist(train_batches)

  # Generate L random permutations
  permutations <- replicate(num_permutations, sample(all_indices), simplify = FALSE)

  # Store results for each permutation
  results <- vector("list", length = num_permutations)

  # Check for divisibility condition
  if (length(all_indices) %% (nb_batch - 1) != 0) {
    stop("Error: The number of individuals in the training batches is not divisible by T-1.")
  }

  for (l in seq_len(num_permutations)) {
    # Current permutation of indices
    current_permutation <- permutations[[l]]

    # Append indices from the evaluation batch (batch T)
    evaluation_indices <- batches[[nb_batch]]  # Indices for batch T
    current_permutation <- c(current_permutation, evaluation_indices)

    # Assign individuals to batches
    batch_indices <- rep(1:nb_batch, each = batch_size)

    # Sort individuals into batch assignment order
    batch_assignment <- integer(n) # Initialize assignment vector for all individuals
    batch_assignment[current_permutation] <- batch_indices

    # Run CRAM experiment with permuted batches
    results[[l]] <- cram_policy(
      X = X, D = D, Y = Y, batch = batch_assignment,
      model_type = model_type, learner_type = learner_type,
      baseline_policy = baseline_policy,
      parallelize_batch = parallelize_batch, model_params = model_params,
      custom_fit = custom_fit, custom_predict = custom_predict, alpha=alpha
    )
  }

  # Extract metrics from each permutation result
  delta_estimates <- sapply(results, function(res) res$raw_results[res$raw_results$Metric == "Delta Estimate", "Value"])
  policy_values <- sapply(results, function(res) res$raw_results[res$raw_results$Metric == "Policy Value Estimate", "Value"])

  avg_delta_estimate <- mean(delta_estimates)
  var_delta_estimate <- var(delta_estimates)
  avg_policy_value <- mean(policy_values)
  var_policy_value <- var(policy_values)

  # Create a summary table
  summary_table <- data.frame(
    Metric = c("Average Delta Estimate", "Delta Estimate Variance",
               "Average Policy Value", "Policy Value Variance"),
    Value = round(c(avg_delta_estimate, var_delta_estimate, avg_policy_value, var_policy_value), 5)
  )

  # Create an interactive table
  interactive_table <- datatable(
    summary_table,
    options = list(pageLength = 5),
    caption = "Averaged CRAM Results"
  )

  # Return results as a list
  return(list(
    summary_table = summary_table,
    interactive_table = interactive_table
  ))
}
