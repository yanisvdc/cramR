path <- "C:/Users/yanis/OneDrive/Documents/"

# Load functions
source(file.path(path, "cram_generate_data.R"))
source(file.path(path, "cram_learning.R"))
source(file.path(path, "cram_estimate.R"))
source(file.path(path, "cram_variance_estimator.R"))
source(file.path(path, "cram_policy_value_estimator.R"))



# Combined simulation function with empirical bias calculation
cram_simulation <- function(X, dgp_D = function(Xi) rbinom(1, 1, 0.5), dgp_Y, batch, nb_simulations, nb_simulations_truth, 
                            model_type = "Causal Forest", learner_type = "ridge", alpha=0.05) {
  
  # Check that nb_simulations_truth is greater than nb_simulations
  if (nb_simulations_truth <= nb_simulations) {
    stop("nb_simulations_truth must be greater than nb_simulations")
  }
  
  # Initialize lists to store results
  result_sim <- list()        # For storing detailed results of nb_simulations
  result_extra_sim <- list()   # For storing only delta_estimate from nb_simulations to nb_simulations_truth
  
  for (i in 1:nb_simulations_truth) {
    # Step 1: Row-wise bootstrap of X
    X_boot <- X[sample(1:nrow(X), nrow(X), replace = TRUE), ]
    
    # Step 2: Generate D for each individual using dgp_D function
    D <- sapply(1:nrow(X_boot), function(j) dgp_D(X_boot[j, ]))
    
    # Step 3: Generate Y for each individual using dgp_Y function
    Y <- sapply(1:nrow(X_boot), function(j) dgp_Y(D[j], X_boot[j, ]))
    
    # Step 4: Run the cram learning process to get policies and batch indices
    learning_result <- cram_learning(X_boot, D, Y, batch, model_type = model_type, learner_type = learner_type)
    policies <- learning_result$policies
    batch_indices <- learning_result$batch_indices
    final_policy_model <- learning_result$final_policy_model
    nb_batch <- length(batch_indices)
    
    # Step 5: Run the cram estimator using the policies and batch indices
    delta_estimate <- cram_estimator(Y, D, policies, batch_indices)
    
    if (i <= nb_simulations) {
      # Step 6: Calculate the proportion of treated individuals under the final policy
      final_policy <- policies[, nb_batch + 1]
      proportion_treated <- mean(final_policy)
      
      # Step 7: Estimate the standard error of delta_estimate using cram_variance_estimator
      asymptotic_variance <- cram_variance_estimator(Y, D, policies, batch_indices)
      asymptotic_sd <- sqrt(asymptotic_variance)
      standard_error <- asymptotic_sd / sqrt(nb_batch)
      
      # Step 8: Compute the 95% confidence interval for delta_estimate
      z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
      ci_lower <- delta_estimate - z_value * standard_error
      ci_upper <- delta_estimate + z_value * standard_error
      confidence_interval <- c(ci_lower, ci_upper)
      
      # Step 9: Estimate policy value
      policy_value_estimate <- cram_policy_value_estimator(Y, D, policies, batch_indices)
      
      result_sim[[i]] <- list(
        final_policy_model = final_policy_model,
        proportion_treated = proportion_treated,
        delta_estimate = delta_estimate,
        standard_error = standard_error,
        confidence_interval = confidence_interval,
        policy_value_estimate = policy_value_estimate
      )
      
    } else {
      # Only store delta_estimate in result_extra_sim for simulations beyond nb_simulations
      result_extra_sim[[i - nb_simulations]] <- delta_estimate
    }
  }
  
  # Calculate average proportion_treated, average delta_estimate and average standard_error in result_sim
  avg_delta_estimate <- mean(sapply(result_sim, function(res) res$delta_estimate))
  avg_standard_error <- mean(sapply(result_sim, function(res) res$standard_error))
  avg_proportion_treated <- mean(sapply(result_sim, function(res) res$proportion_treated))
  avg_policy_value_estimate<- mean(sapply(result_sim, function(res) res$policy_value_estimate))
  
  # Calculate the true_value as the average delta_estimate across both result_sim and result_extra_sim
  all_delta_estimates <- c(avg_delta_estimate, unlist(result_extra_sim))
  true_value <- mean(all_delta_estimates)
  
  # Calculate empirical bias
  empirical_bias <- avg_delta_estimate - true_value
  
  # Calculate empirical coverage of the confidence interval
  coverage_count <- sum(sapply(result_sim, function(res) {
    res$confidence_interval[1] <= true_value && res$confidence_interval[2] >= true_value
  }))
  empirical_coverage <- coverage_count / nb_simulations  # Proportion of CIs containing true_value
  
  # Return the final results
  result <- list(
    avg_proportion_treated = avg_proportion_treated,
    average_delta_estimate = avg_delta_estimate,
    average_standard_error = avg_standard_error,
    empirical_bias = empirical_bias,
    empirical_coverage = empirical_coverage,
    avg_policy_value_estimate = avg_policy_value_estimate
  )
  
  return(result)
}

# Example usage of cram_simulation

set.seed(123)  # For reproducibility

# Define individualized data-generating processes

# dgp_D: Assign treatment with a Bernoulli(0.5) probability based on individual's covariates Xi
dgp_D <- function(Xi) {
  return(rbinom(1, 1, 0.5))  # Each individual has a 50% chance of treatment
}

# dgp_Y: outcome generation as a function of individual D and X with noise
# Define dgp_Y with heterogeneous treatment effects
dgp_Y <- function(d, x) {
  # Define theta based on individual's covariates
  theta <- ifelse(
    x["binary"] == 1 & x["discrete"] <= 2,  # Group 1: High benefit
    1,
    ifelse(x["binary"] == 0 & x["discrete"] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect (small positive or negative)
  )
  
  # Define outcome Y with treatment effect and noise
  y <- d * (theta + rnorm(1, mean = 0, sd = 1)) + 
    (1 - d) * rnorm(1)  # Outcome influenced by treatment and noise for untreated
  
  # Ensure Y has no names by converting it to an unnamed numeric value
  return(unname(y))
}

# Load the input data (replace generate_data function with actual X matrix)
n <- 1000  # Number of samples
X <- generate_data(n)$X  # Load or generate X

# Number of simulations and batches
nb_simulations <- 2
nb_simulations_truth <- 4  # nb_simulations_truth must be greater than nb_simulations
batch <- 20  

# Run cram_simulation
simulation_results <- cram_simulation(X, dgp_D, dgp_Y, batch, nb_simulations, nb_simulations_truth, 
                                      model_type = "Causal Forest", learner_type = "ridge")

# Print the final summarized results
print(simulation_results)
