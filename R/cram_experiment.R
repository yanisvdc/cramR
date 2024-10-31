path <- "C:/Users/yanis/OneDrive/Documents/"

# Load functions
source(file.path(path, "cram_generate_data.R"))
source(file.path(path, "cram_learning.R"))
source(file.path(path, "cram_estimate.R"))
source(file.path(path, "cram_variance_estimator.R"))
source(file.path(path, "cram_policy_value_estimator.R"))


# Combined experiment function
cram_experiment <- function(X, D, Y, batch, model_type = "Causal Forest", learner_type = "ridge", alpha=0.05) {
  # Step 1: Run the cram learning process to get policies and batch indices
  learning_result <- cram_learning(X, D, Y, batch, model_type = model_type, learner_type = learner_type)
  policies <- learning_result$policies
  batch_indices <- learning_result$batch_indices
  final_policy_model <- learning_result$final_policy_model
  nb_batch <- length(batch_indices)
  
  # Step 2: Calculate the proportion of treated individuals under the final policy
  final_policy <- policies[, nb_batch + 1]  # Extract the final policy
  proportion_treated <- mean(final_policy)  # Proportion of treated individuals
  
  # Step 3: Run the cram estimator using the policies and batch indices
  delta_estimate <- cram_estimator(Y, D, policies, batch_indices)
  
  # Step 4: Estimate the standard error of delta_estimate using cram_variance_estimator
  asymptotic_variance <- cram_variance_estimator(Y, D, policies, batch_indices)
  asymptotic_sd <- sqrt(asymptotic_variance)  # v_T, the asymptotic standard deviation
  standard_error <- asymptotic_sd / sqrt(nb_batch)  # Standard error based on T (number of batches)
  
  # Step 5: Compute the 95% confidence interval for delta_estimate
  z_value <- qnorm(1 - alpha / 2)  # Critical z-value based on the alpha level
  ci_lower <- delta_estimate - z_value * standard_error
  ci_upper <- delta_estimate + z_value * standard_error
  confidence_interval <- c(ci_lower, ci_upper)
  
  # Step 6: Estimate policy value
  policy_value_estimate <- cram_policy_value_estimator(Y, D, policies, batch_indices)
  
  
  return(list(final_policy_model = final_policy_model,
              proportion_treated = proportion_treated,
              delta_estimate = delta_estimate, 
              standard_error = standard_error,
              confidence_interval = confidence_interval,
              policy_value_estimate = policy_value_estimate))
  
}



# Example usage of cram_experiment
set.seed(123)  # For reproducibility

n <- 1000  # Number of samples
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y


# Number of batches
batch <- 20  

# Run cram_experiment
estimate <- cram_experiment(X, D, Y, batch)
print(estimate)