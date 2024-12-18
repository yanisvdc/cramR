# library(parallel)
# Load necessary libraries
# install.packages(c("doParallel", "foreach"))
# install.packages("data.table")

# Set JIT level to 0 to disable JIT compilation
compiler::enableJIT(0)

library(data.table)
library(doParallel)
library(foreach)
library(grf)      # Load grf package for causal_forest
library(glmnet)   # Load glmnet package
library(keras)    # Load keras package
library(data.table)


path <- "C:/Users/yanis/OneDrive/Documents/cramR/R"

# Load functions
source(file.path(path, "cram_generate_data.R"))
source(file.path(path, "cram_experiment.R"))
source(file.path(path, "cram_simulation.R"))

source(file.path(path, "cram_learning.R"))

source(file.path(path, "cram_estimate.R"))
source(file.path(path, "cram_variance_estimator.R"))
source(file.path(path, "cram_policy_value_estimator.R"))
source(file.path(path, "cram_variance_estimator_policy_value.R"))

source(file.path(path, "set_model.R"))
source(file.path(path, "fit_model.R"))
source(file.path(path, "model_predict.R"))
source(file.path(path, "validate_params.R"))
source(file.path(path, "validate_params_fnn.R"))
source(file.path(path, "test_func.R"))

# Example usage of CRAM EXPERIMENT
set.seed(123)

## Generate data
n <- 1000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

## Parameters
batch <- 20
model_type <- "m_learner" # causal_forest, s_learner, m_learner
learner_type <- "ridge" # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- TRUE
model_params <- NULL

## Run cram_experiment
experiment_results <- cram_experiment(X, D, Y, batch, model_type = model_type,
         learner_type = learner_type, alpha=alpha, baseline_policy = baseline_policy,
         parallelize_batch = parallelize_batch, model_params = model_params)

print(experiment_results)

# --------------------------------------------------------------------------------------------
# Load necessary libraries if not already loaded
# library(causalforest) # Load your specific model library if needed

# Example usage of CRAM SIMULATION
set.seed(123)

## Obtain reference dataset X and define the data generation processes for D and Y
## dgp_D (resp. dgp_Y) takes individual-level data Xi (resp. Xi and Di) as inputs
n <- 1000
data <- generate_data(n)
X <- data$X
# dgp_D <- function(Xi) {
#   return(rbinom(1, 1, 0.5))
# }
dgp_D <- function(X) {
  # Generate a vector of binary treatment assignments for all individuals at once
  return(rbinom(nrow(X), 1, 0.5))
}
# Vectorized dgp_Y for all individuals
dgp_Y <- function(D, X) {
  # Calculate theta for each individual based on the covariates
  theta <- ifelse(
    X[, binary] == 1 & X[, discrete] <= 2,  # Group 1: High benefit
    1,
    ifelse(X[, binary] == 0 & X[, discrete] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect
  )

  # Generate Y values with treatment effect and random noise
  Y <- D * (theta + rnorm(length(D), mean = 0, sd = 1)) +
    (1 - D) * rnorm(length(D))  # Outcome for untreated

  return(Y)
}

# Generate batches with fixed last batch
generate_batches <- function(X, num_batches) {
  n <- nrow(X)
  batch_size <- n %/% num_batches
  remainder <- n %% num_batches

  # Create initial indices for batches
  indices <- sample(n)
  batches <- split(indices, rep(1:num_batches, each = batch_size, length.out = n))

  return(batches)
}

# Average results over 100 runs
average_cram_simulation <- function(X, dgp_D, dgp_Y, batch, nb_simulations,
                                    nb_simulations_truth, model_type, learner_type,
                                    alpha, baseline_policy, model_params, num_runs = 100) {
  # Placeholder for accumulating results
  results <- vector("list", num_runs)

  # Constant last batch
  fixed_last_batch <- generate_batches(X, batch)[[batch]]

  for (run in seq_len(num_runs)) {
    # Permute indices for first n-1 batches
    permuted_batches <- generate_batches(X, batch - 1)
    permuted_batches[[batch]] <- fixed_last_batch  # Keep the last batch fixed

    # Combine permuted batches into a single list of indices
    full_batch <- lapply(seq_along(permuted_batches), function(i) {
      permuted_batches[[i]]
    })

    # Run cram_simulation with permuted batches
    results[[run]] <- cram_simulation(X, dgp_D, dgp_Y, full_batch,
                                      nb_simulations, nb_simulations_truth,
                                      model_type, learner_type, alpha,
                                      baseline_policy, model_params)
  }

  # Average the results
  averaged_results <- Reduce(function(a, b) {
    Map(`+`, a, b)
  }, results)
  averaged_results <- lapply(averaged_results, function(x) x / num_runs)

  return(averaged_results)
}

# Run the function
set.seed(123)  # Ensure reproducibility
num_runs <- 100
batch <- 20

# Generate reference data
n <- 1000
data <- generate_data(n)
X <- data$X

# Parameters
nb_simulations <- 2
nb_simulations_truth <- 4
model_type <- "causal_forest"
learner_type <- "NULL"
baseline_policy <- as.list(rep(0, nrow(X)))
alpha <- 0.05
model_params <- list(num.trees = 100)

# Average cram_simulation over 100 runs
averaged_results <- average_cram_simulation(X, dgp_D, dgp_Y, batch, nb_simulations,
                                            nb_simulations_truth, model_type,
                                            learner_type, alpha, baseline_policy,
                                            model_params, num_runs)

# Print the averaged results
print(averaged_results)





# -----------------------------------------------
# Example usage of CRAM SIMULATION
set.seed(123)

## Obtain reference dataset X and define the data generation processes for D and Y
## dgp_D (resp. dgp_Y) takes individual-level data Xi (resp. Xi and Di) as inputs
n <- 1000
data <- generate_data(n)
X <- data$X
# dgp_D <- function(Xi) {
#   return(rbinom(1, 1, 0.5))
# }
dgp_D <- function(X) {
  # Generate a vector of binary treatment assignments for all individuals at once
  return(rbinom(nrow(X), 1, 0.5))
}
# Vectorized dgp_Y for all individuals
dgp_Y <- function(D, X) {
  # Calculate theta for each individual based on the covariates
  theta <- ifelse(
    X[, binary] == 1 & X[, discrete] <= 2,  # Group 1: High benefit
    1,
    ifelse(X[, binary] == 0 & X[, discrete] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect
  )

  # Generate Y values with treatment effect and random noise
  Y <- D * (theta + rnorm(length(D), mean = 0, sd = 1)) +
    (1 - D) * rnorm(length(D))  # Outcome for untreated

  return(Y)
}

## Parameters
batch <- 20
nb_simulations <- 2
nb_simulations_truth <- 4  # nb_simulations_truth must be greater than nb_simulations
model_type <- "m_learner" # causal_forest, s_learner, m_learner
learner_type <- "ridge" # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL

## Run cram_experiment
# install.packages("profvis")
# library(profvis)
print(Sys.time())
simulation_results <- cram_simulation(X, dgp_D, dgp_Y, batch,
                          nb_simulations, nb_simulations_truth,
                          model_type, learner_type,
                          alpha, baseline_policy, model_params=model_params)
print(Sys.time())
print(simulation_results)



## Test Cram Learning

# Example usage of CRAM EXPERIMENT
set.seed(123)

## Generate data
n <- 1000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

## Parameters
batch <- 20
model_type <- "m_learner" # causal_forest, s_learner, m_learner
learner_type <- "ridge" # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- TRUE
model_params <- NULL


learning_result <- cram_learning(X, D, Y, batch, model_type = model_type,
                                 learner_type = learner_type, baseline_policy = baseline_policy,
                                 parallelize_batch = parallelize_batch, model_params = model_params)



print(learning_result)
# policies <- learning_result$policies
# batch_indices <- learning_result$batch_indices
# final_policy_model <- learning_result$final_policy_model




