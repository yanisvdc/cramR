# Combined experiment function
cram_experiment <- function(X, D, Y, batch, model_type = "Causal Forest",
                            learner_type = "ridge", alpha=0.05, baseline_policy = NULL) {

  # Step 0: Set default baseline_policy if NULL
  if (is.null(baseline_policy)) {
    print("Baseline policy is NULL: policy value and policy value difference are the same.")
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
    # Check if baseline_policy contains only zeros
    if (all(sapply(baseline_policy, function(x) x == 0))) {
      print("Baseline policy contains only zeros: policy value and policy value difference are the same.")
    }
  }

  # Step 1: Run the cram learning process to get policies and batch indices
  learning_result <- cram_learning(X, D, Y, batch, model_type = model_type,
                                   learner_type = learner_type, baseline_policy = baseline_policy)
  policies <- learning_result$policies
  batch_indices <- learning_result$batch_indices
  final_policy_model <- learning_result$final_policy_model
  nb_batch <- length(batch_indices)

  # Step 2: Calculate the proportion of treated individuals under the final policy
  final_policy <- policies[, nb_batch + 1]  # Extract the final policy
  proportion_treated <- mean(final_policy)  # Proportion of treated individuals

  # Step 3: Estimate delta
  delta_estimate <- cram_estimator(Y, D, policies, batch_indices)

  # Step 4: Estimate the standard error of delta_estimate using cram_variance_estimator
  delta_asymptotic_variance <- cram_variance_estimator(Y, D, policies, batch_indices)
  delta_asymptotic_sd <- sqrt(delta_asymptotic_variance)  # v_T, the asymptotic standard deviation
  delta_standard_error <- delta_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

  # Step 5: Compute the 95% confidence interval for delta_estimate
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  delta_ci_lower <- delta_estimate - z_value * delta_standard_error
  delta_ci_upper <- delta_estimate + z_value * delta_standard_error
  delta_confidence_interval <- c(delta_ci_lower, delta_ci_upper)

  # Step 6: Estimate policy value
  policy_value_estimate <- cram_policy_value_estimator(Y, D,
                                                       policies,
                                                       batch_indices)

  # Step 7: Estimate the standard error of policy_value_estimate using cram_variance_estimator
  ## same as delta, but enforcing a null baseline policy
  null_baseline <- as.list(rep(0, nrow(X)))
  policies_with_null_baseline <- policies
  policies_with_null_baseline[, 1] <- unlist(null_baseline)  # Set the first column to baseline policy

  policy_value_asymptotic_variance <- cram_variance_estimator(Y, D,
                                                              policies_with_null_baseline,
                                                              batch_indices)
  policy_value_asymptotic_sd <- sqrt(policy_value_asymptotic_variance)  # w_T, the asymptotic standard deviation
  policy_value_standard_error <- policy_value_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

  # Step 8: Compute the 95% confidence interval for policy_value_estimate
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  policy_value_ci_lower <- policy_value_estimate - z_value * policy_value_standard_error
  policy_value_ci_upper <- policy_value_estimate + z_value * policy_value_standard_error
  policy_value_confidence_interval <- c(policy_value_ci_lower, policy_value_ci_upper)






  return(list(final_policy_model = final_policy_model,
              proportion_treated = proportion_treated,
              delta_estimate = delta_estimate,
              delta_standard_error = delta_standard_error,
              delta_confidence_interval = delta_confidence_interval,
              policy_value_estimate = policy_value_estimate,
              policy_value_standard_error = policy_value_standard_error,
              policy_value_confidence_interval = policy_value_confidence_interval))

}
