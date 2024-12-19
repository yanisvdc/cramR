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
cram_simulation <- function(X, dgp_D = function(X) rbinom(1, 1, 0.5), dgp_Y, batch,
                            nb_simulations, nb_simulations_truth, sample_size,
                            model_type = "causal_forest", learner_type = "ridge",
                            alpha=0.05, baseline_policy = NULL,
                            parallelize_batch = FALSE, model_params = list()) {

  # Step 0: Set default baseline_policy if NULL
  if (is.null(baseline_policy)) {
    print("Baseline policy is NULL: policy value and policy value difference are the same.")
    baseline_policy <- as.list(rep(0, nrow(X)))  # Creates a list of zeros with the same length as X
  } else {
    # Validate baseline_policy if provided
    if (!is.list(baseline_policy)) {
      stop("Error: baseline_policy must be a list.")
    }
    if (length(baseline_policy) != sample_size) {
      stop("Error: baseline_policy length must match the sample size.")
    }
    if (!all(sapply(baseline_policy, is.numeric))) {
      stop("Error: baseline_policy must contain numeric values only.")
    }
  }

  # Check that nb_simulations_truth is greater than nb_simulations
  if (nb_simulations_truth <= nb_simulations) {
    stop("nb_simulations_truth must be greater than nb_simulations")
  }

  # Precompute bootstrap indices for all simulations
  X_size <- nrow(X)
  total_samples <- nb_simulations * sample_size
  sampled_indices <- sample(1:X_size, size = total_samples, replace = TRUE)

  # Convert the smaller matrix X to a data.table
  X_dt <- as.data.table(X)

  # Use sampled_indices on the data.table to create big_X
  sim_ids <- rep(1:nb_simulations, each = sample_size)  # Simulation IDs
  big_X <- X_dt[sampled_indices]  # Subset the data.table using the sampled indices
  big_X[, sim_id := sim_ids]  # Add simulation IDs

  # Add D and Y columns by applying dgp_D and dgp_Y
  big_X[, D := dgp_D(.SD), by = sim_id]
  big_X[, Y := dgp_Y(D, .SD), by = sim_id]

  # Set key for fast grouping and operations
  setkey(big_X, sim_id)

  # Initialize lists to store results
  result_sim <- vector("list", nb_simulations)   # For storing detailed results of nb_simulations
  result_extra_sim <- vector("list", nb_simulations_truth - nb_simulations)   # For storing only delta_estimate from nb_simulations to nb_simulations_truth

  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level

  cram_results <- big_X[, {
    # Dynamically select all columns except Y and D for covariates
    X_matrix <- as.matrix(.SD[, !c("Y", "D"), with = FALSE])  # Exclude Y and D dynamically

    # Extract D and Y for the current group
    D_slice <- D
    Y_slice <- Y

    # Run the cram_learning function
    learning_result <- cram_learning(
      X_matrix,
      D_slice,
      Y_slice,
      batch,
      model_type = model_type,
      learner_type = learner_type,
      baseline_policy = baseline_policy,
      parallelize_batch = parallelize_batch,
      model_params = model_params
    )

    policies <- learning_result$policies
    batch_indices <- learning_result$batch_indices
    final_policy_model <- learning_result$final_policy_model
    nb_batch <- length(batch_indices)

    # Step 5: Estimate delta
    delta_estimate <- cram_estimator(Y_slice, D_slice, policies, batch_indices)

    # Step 5': Estimate policy value
    policy_value_estimate <- cram_policy_value_estimator(Y_slice, D_slice,
                                                         policies,
                                                         batch_indices)

    # Step 5 TRUE: Estimate true delta and true policy value
    true_results <- big_X[, {
      # Extract D and Y for the current group
      D_slice <- D
      Y_slice <- Y

      true_delta_estimate <- cram_estimator(Y_slice, D_slice, policies, batch_indices)
      true_policy_value_estimate <- cram_policy_value_estimator(Y_slice, D_slice,
                                                           policies,
                                                           batch_indices)

      .(
        true_delta_estimate,
        true_policy_value_estimate
      )

    }, by = sim_id]

    true_delta <- mean(true_results$true_delta_estimate)
    true_policy_value <- mean(true_results$true_policy_value_estimate)

    # prediction_error_delta <- delta_estimate - true_delta
    # prediction_error_policy_value <- policy_value_estimate - true_policy_value


    # Step 6: Calculate the proportion of treated individuals under the final policy
    final_policy <- policies[, nb_batch + 1]
    proportion_treated <- mean(final_policy)

    # Step 7: Estimate the standard error of delta_estimate using cram_variance_estimator
    delta_asymptotic_variance <- cram_variance_estimator(Y_slice, D_slice, policies, batch_indices)
    delta_asymptotic_sd <- sqrt(delta_asymptotic_variance)  # v_T, the asymptotic standard deviation
    delta_standard_error <- delta_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

    # Step 8: Compute the 95% confidence interval for delta_estimate
    delta_ci_lower <- delta_estimate - z_value * delta_standard_error
    delta_ci_upper <- delta_estimate + z_value * delta_standard_error
    delta_confidence_interval <- c(delta_ci_lower, delta_ci_upper)

    # Step 9: Estimate the standard error of policy_value_estimate using cram_variance_estimator_policy_value
    policy_value_asymptotic_variance <- cram_variance_estimator_policy_value(Y_slice,
                                                                             D_slice,
                                                                             policies,
                                                                             batch_indices)
    policy_value_asymptotic_sd <- sqrt(policy_value_asymptotic_variance)  # w_T, the asymptotic standard deviation
    policy_value_standard_error <- policy_value_asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)

    # Step 10: Compute the 95% confidence interval for policy_value_estimate
    policy_value_ci_lower <- policy_value_estimate - z_value * policy_value_standard_error
    policy_value_ci_upper <- policy_value_estimate + z_value * policy_value_standard_error
    policy_value_confidence_interval <- c(policy_value_ci_lower, policy_value_ci_upper)

    # Assign results as new columns
    .(
      # final_policy_model = list(final_policy_model),
      proportion_treated = proportion_treated,
      delta_estimate = delta_estimate,
      delta_asymptotic_variance = delta_asymptotic_variance,
      delta_standard_error = delta_standard_error,
      delta_ci_lower = delta_ci_lower,
      delta_ci_upper = delta_ci_upper,
      policy_value_estimate = policy_value_estimate,
      policy_value_asymptotic_variance = policy_value_asymptotic_variance,
      policy_value_standard_error = policy_value_standard_error,
      policy_value_ci_lower = policy_value_ci_lower,
      policy_value_ci_upper = policy_value_ci_upper,
      true_delta = true_delta,
      true_policy_value = true_policy_value
    )

  }, by = sim_id]

  # Filter cram_results for the first nb_simulations rows
  sim_results <- cram_results

  # Calculate averages for the desired columns
  avg_proportion_treated <- mean(sim_results$proportion_treated)
  avg_delta_estimate <- mean(sim_results$delta_estimate)
  avg_delta_standard_error <- mean(sim_results$delta_standard_error)
  avg_policy_value_estimate <- mean(sim_results$policy_value_estimate)
  avg_policy_value_standard_error <- mean(sim_results$policy_value_standard_error)

  prediction_error_delta <- sim_results$delta_estimate - sim_results$true_delta
  prediction_error_policy_value <- sim_results$policy_value_estimate - sim_results$true_policy_value

  # Calculate empirical bias
  delta_empirical_bias <- mean(prediction_error_delta)
  policy_value_empirical_bias <- mean(prediction_error_policy_value)

  # Bias for variance estimate
  true_asymp_var_delta <- var(prediction_error_delta)
  true_asymp_var_policy_value <- var(prediction_error_policy_value)

  var_delta_empirical_bias <- mean(sim_results$delta_asymptotic_variance - true_asymp_var_delta)
  var_policy_value_empirical_bias <- mean(sim_results$policy_value_asymptotic_variance - true_asymp_var_policy_value)

  # Calculate empirical coverage of the confidence interval using sim_results
  delta_coverage_count <- sum(
    sim_results$delta_ci_lower <= sim_results$true_delta & sim_results$delta_ci_upper >= sim_results$true_delta
  )
  delta_empirical_coverage <- delta_coverage_count / nb_simulations  # Proportion of CIs containing true_value

  # Calculate empirical coverage of the policy value confidence interval using sim_results
  policy_value_coverage_count <- sum(
    sim_results$policy_value_ci_lower <= sim_results$true_policy_value & sim_results$policy_value_ci_upper >= sim_results$true_policy_value
  )
  policy_value_empirical_coverage <- policy_value_coverage_count / nb_simulations  # Proportion of CIs containing true_value

  # Return the final results
  result <- list(
    avg_proportion_treated = avg_proportion_treated,
    avg_delta_estimate = avg_delta_estimate,
    avg_delta_standard_error = avg_delta_standard_error,
    delta_empirical_bias = delta_empirical_bias,
    delta_empirical_coverage = delta_empirical_coverage,
    var_delta_empirical_bias = var_delta_empirical_bias,
    avg_policy_value_estimate = avg_policy_value_estimate,
    avg_policy_value_standard_error = avg_policy_value_standard_error,
    policy_value_empirical_bias = policy_value_empirical_bias,
    policy_value_empirical_coverage = policy_value_empirical_coverage,
    var_policy_value_empirical_bias = var_policy_value_empirical_bias
  )

  return(result)

}
