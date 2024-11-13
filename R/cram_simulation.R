#' CRAM Simulation with Empirical Bias Calculation
#'
#' This function simulates the estimation of treatment effects in a cumulative randomized assignment model (CRAM) setup. It uses bootstrapping to resample data, applies treatment and outcome generation functions, and calculates empirical bias and coverage rates for delta and policy value estimates.
#'
#' @param X A matrix or data frame of covariates for each sample.
#' @param dgp_D A function to generate binary treatment assignments for each sample. Defaults to \code{function(Xi) rbinom(1, 1, 0.5)} for random assignment.
#' @param dgp_Y A function to generate the outcome variable for each sample given the treatment and covariates.
#' @param batch Either an integer specifying the number of batches (which will be created by random sampling) or a list/vector providing specific batch indices.
#' @param nb_simulations The number of main simulations to run. Full results are stored for each of these simulations.
#' @param nb_simulations_truth The total number of simulations to run, which should be greater than \code{nb_simulations}. Only the delta estimates are stored for simulations beyond \code{nb_simulations} up to \code{nb_simulations_truth}.
#' @param model_type The model type for policy learning. Options include \code{"Causal Forest"}, \code{"S-learner"}, and \code{"M-learner"}. Default is \code{"Causal Forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"FNN"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param alpha Significance level for confidence intervals. Default is 0.05 (95% confidence).
#' @param baseline_policy A list providing the baseline policy (binary 0 or 1) for each sample. If \code{NULL}, defaults to a list of zeros with the same length as the number of samples in \code{X}.
#' @return A list containing:
#'   \item{avg_proportion_treated}{The average proportion of treated individuals across simulations.}
#'   \item{avg_delta_estimate}{The average delta estimate across simulations.}
#'   \item{avg_delta_standard_error}{The average standard error of delta estimates.}
#'   \item{delta_empirical_bias}{The empirical bias of delta estimates.}
#'   \item{delta_empirical_coverage}{The empirical coverage of delta confidence intervals.}
#'   \item{avg_policy_value_estimate}{The average policy value estimate across simulations.}
#'   \item{avg_policy_value_standard_error}{The average standard error of policy value estimates.}
#'   \item{policy_value_empirical_bias}{The empirical bias of policy value estimates.}
#'   \item{policy_value_empirical_coverage}{The empirical coverage of policy value confidence intervals.}
#' @examples
#' # Define data generation process (DGP) functions
#' dgp_D <- function(Xi) rbinom(1, 1, 0.5)
#' dgp_Y <- function(D, Xi) D * rnorm(1, mean = 1) + (1 - D) * rnorm(1, mean = 0)
#'
#' # Example data
#' X_data <- matrix(rnorm(100 * 5), nrow = 100, ncol = 5)  # 100 samples, 5 features
#' nb_simulations <- 10
#' nb_simulations_truth <- 20
#' batch <- 3
#'
#' # Perform CRAM simulation
#' result <- cram_simulation(X = X_data, dgp_D = dgp_D, dgp_Y = dgp_Y, batch = batch,
#'                           nb_simulations = nb_simulations, nb_simulations_truth = nb_simulations_truth)
#'
#' # Access results
#' result$avg_delta_estimate
#' result$delta_empirical_bias
#' @seealso \code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{keras_model_sequential}}
#' @export

# Combined simulation function with empirical bias calculation
cram_simulation <- function(X, dgp_D = function(Xi) rbinom(1, 1, 0.5), dgp_Y, batch,
                            nb_simulations, nb_simulations_truth,
                            model_type = "Causal Forest", learner_type = "ridge",
                            alpha=0.05, baseline_policy = NULL) {

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

  # Check that nb_simulations_truth is greater than nb_simulations
  if (nb_simulations_truth <= nb_simulations) {
    stop("nb_simulations_truth must be greater than nb_simulations")
  }

  # Initialize lists to store results
  result_sim <- vector("list", nb_simulations)   # For storing detailed results of nb_simulations
  result_extra_sim <- vector("list", nb_simulations_truth - nb_simulations)   # For storing only delta_estimate from nb_simulations to nb_simulations_truth

  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  null_baseline <- as.list(rep(0, nrow(X)))

  for (i in 1:nb_simulations_truth) {
    # Step 1: Row-wise bootstrap of X
    X_boot <- X[sample(1:nrow(X), nrow(X), replace = TRUE), ]

    # # Step 2: Generate D for each individual using dgp_D function
    D <- vapply(1:nrow(X_boot), function(j) dgp_D(X_boot[j, ]), numeric(1))

    # # Step 3: Generate Y for each individual using dgp_Y function
    Y <- vapply(1:nrow(X_boot), function(j) dgp_Y(D[j], X_boot[j, ]), numeric(1))

    # Step 4: Run the cram learning process to get policies and batch indices
    learning_result <- cram_learning(X_boot, D, Y, batch, model_type = model_type,
                                     learner_type = learner_type, baseline_policy = baseline_policy)

    policies <- learning_result$policies
    batch_indices <- learning_result$batch_indices
    final_policy_model <- learning_result$final_policy_model
    nb_batch <- length(batch_indices)

    # Step 5: Estimate delta
    delta_estimate <- cram_estimator(Y, D, policies, batch_indices)

    # Step 5': Estimate policy value
    policy_value_estimate <- cram_policy_value_estimator(Y, D,
                                                         policies,
                                                         batch_indices)

    if (i <= nb_simulations) {
      # Step 6: Calculate the proportion of treated individuals under the final policy
      final_policy <- policies[, nb_batch + 1]
      proportion_treated <- mean(final_policy)

      # Step 7: Estimate the standard error of delta_estimate using cram_variance_estimator
      delta_asymptotic_variance <- cram_variance_estimator(Y, D, policies, batch_indices)
      delta_asymptotic_sd <- sqrt(delta_asymptotic_variance)  # v_T, the asymptotic standard deviation
      delta_standard_error <- delta_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

      # Step 8: Compute the 95% confidence interval for delta_estimate
      delta_ci_lower <- delta_estimate - z_value * delta_standard_error
      delta_ci_upper <- delta_estimate + z_value * delta_standard_error
      delta_confidence_interval <- c(delta_ci_lower, delta_ci_upper)

      # Step 9: Estimate the standard error of policy_value_estimate using cram_variance_estimator
      ## same as delta, but enforcing a null baseline policy
      policies_with_null_baseline <- policies
      policies_with_null_baseline[, 1] <- unlist(null_baseline)  # Set the first column to baseline policy

      policy_value_asymptotic_variance <- cram_variance_estimator(Y, D,
                                                                  policies_with_null_baseline,
                                                                  batch_indices)
      policy_value_asymptotic_sd <- sqrt(policy_value_asymptotic_variance)  # w_T, the asymptotic standard deviation
      policy_value_standard_error <- policy_value_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

      # Step 10: Compute the 95% confidence interval for policy_value_estimate
      policy_value_ci_lower <- policy_value_estimate - z_value * policy_value_standard_error
      policy_value_ci_upper <- policy_value_estimate + z_value * policy_value_standard_error
      policy_value_confidence_interval <- c(policy_value_ci_lower, policy_value_ci_upper)


      result_sim[[i]] <- list(
        final_policy_model = final_policy_model,
        proportion_treated = proportion_treated,
        delta_estimate = delta_estimate,
        delta_standard_error = delta_standard_error,
        delta_confidence_interval = delta_confidence_interval,
        policy_value_estimate = policy_value_estimate,
        policy_value_standard_error = policy_value_standard_error,
        policy_value_confidence_interval = policy_value_confidence_interval
      )


    } else {
      # Only store delta_estimate in result_extra_sim for simulations beyond nb_simulations
      result_extra_sim[[i - nb_simulations]] <- list(delta_estimate, policy_value_estimate)
    }
  }

  # Calculate average proportion_treated, average delta_estimate and average standard_error in result_sim
  avg_proportion_treated <- mean(sapply(result_sim, function(res) res$proportion_treated))
  avg_delta_estimate <- mean(sapply(result_sim, function(res) res$delta_estimate))
  avg_delta_standard_error <- mean(sapply(result_sim, function(res) res$delta_standard_error))
  avg_policy_value_estimate <- mean(sapply(result_sim, function(res) res$policy_value_estimate))
  avg_policy_value_standard_error <- mean(sapply(result_sim, function(res) res$policy_value_standard_error))

  # Calculate the true_value of delta as the average delta_estimate across both result_sim and result_extra_sim
  all_delta_estimates <- c(avg_delta_estimate, unlist(lapply(result_extra_sim, `[[`, 1)))
  true_delta <- mean(all_delta_estimates)

  # Calculate the true_value of policy_value as the average polcy_value_estimate across both result_sim and result_extra_sim
  all_policy_value_estimates <- c(avg_policy_value_estimate, unlist(lapply(result_extra_sim, `[[`, 2)))
  true_policy_value <- mean(all_policy_value_estimates)

  # Calculate empirical bias
  delta_empirical_bias <- avg_delta_estimate - true_delta
  policy_value_empirical_bias <- avg_policy_value_estimate - true_policy_value

  # Calculate empirical coverage of the confidence interval
  delta_coverage_count <- sum(sapply(result_sim, function(res) {
    res$delta_confidence_interval[1] <= true_delta && res$delta_confidence_interval[2] >= true_delta
  }))
  delta_empirical_coverage <- delta_coverage_count / nb_simulations  # Proportion of CIs containing true_value

  policy_value_coverage_count <- sum(sapply(result_sim, function(res) {
    res$policy_value_confidence_interval[1] <= true_policy_value && res$policy_value_confidence_interval[2] >= true_policy_value
  }))
  policy_value_empirical_coverage <- policy_value_coverage_count / nb_simulations  # Proportion of CIs containing true_value

  # Return the final results
  result <- list(
    avg_proportion_treated = avg_proportion_treated,
    avg_delta_estimate = avg_delta_estimate,
    avg_delta_standard_error = avg_delta_standard_error,
    delta_empirical_bias = delta_empirical_bias,
    delta_empirical_coverage = delta_empirical_coverage,
    avg_policy_value_estimate = avg_policy_value_estimate,
    avg_policy_value_standard_error = avg_policy_value_standard_error,
    policy_value_empirical_bias = policy_value_empirical_bias,
    policy_value_empirical_coverage = policy_value_empirical_coverage
  )

  return(result)
}
