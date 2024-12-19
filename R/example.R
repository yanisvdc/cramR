# library(parallel)
# Load necessary libraries
# install.packages(c("doParallel", "foreach"))
# install.packages("data.table")

# Set JIT level to 0 to disable JIT compilation
compiler::enableJIT(0)

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
source(file.path(path, "averaged_cram.R"))


## Test Cram Learning

# Example usage of CRAM LEARNING
set.seed(123)

## Generate data
n <- 1000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

## Parameters
batch <- 20
model_type <- "s_learner" # causal_forest, s_learner, m_learner
learner_type <- "fnn" # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL


learning_result <- cram_learning(X, D, Y, batch, model_type = model_type,
                                 learner_type = learner_type, baseline_policy = baseline_policy,
                                 parallelize_batch = parallelize_batch, model_params = model_params,
                                 custom_fit = NULL, custom_predict = NULL)


print(learning_result)
# policies <- learning_result$policies
# batch_indices <- learning_result$batch_indices
# final_policy_model <- learning_result$final_policy_model

# --------------------------------------------------------------------------------------

# Custom X-Learner: Returns only the final model
custom_fit <- function(X, Y, D) {

  # Split the data into treated and control groups
  treated_indices <- which(D == 1)
  control_indices <- which(D == 0)

  X_treated <- X[treated_indices, ]
  Y_treated <- Y[treated_indices]
  X_control <- X[control_indices, ]
  Y_control <- Y[control_indices]

  # Step 1: Fit base models on treated and control groups separately
  model_treated <- cv.glmnet(as.matrix(X_treated), Y_treated, alpha = 0)
  model_control <- cv.glmnet(as.matrix(X_control), Y_control, alpha = 0)

  # Step 2: Compute pseudo-outcomes
  # Predict outcomes for the treated group using the control model
  tau_control <- Y_treated - predict(model_control, as.matrix(X_treated), s = "lambda.min")

  # Predict outcomes for the control group using the treated model
  tau_treated <- predict(model_treated, as.matrix(X_control), s = "lambda.min") - Y_control

  # Step 3: Combine pseudo-outcomes into a single training set
  X_combined <- rbind(X_treated, X_control)
  tau_combined <- c(tau_control, tau_treated)
  weights <- c(rep(1, length(tau_control)), rep(1, length(tau_treated))) # Equal weighting

  # Step 4: Fit a single model on the combined pseudo-outcomes
  final_model <- cv.glmnet(as.matrix(X_combined), tau_combined, alpha = 0, weights = weights)

  return(final_model)
}

# Custom prediction function
custom_predict <- function(model, X_new, D_new) {
  # Use the final model for predictions on new data
  cate <- predict(model, as.matrix(X_new), s = "lambda.min")
  as.numeric(cate) # Return as a numeric vector
}


# Example usage of CRAM LEARNING
set.seed(123)

## Generate data
n <- 10000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

## Parameters
batch <- 20
model_type <- NULL # causal_forest, s_learner, m_learner
learner_type <- NULL # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL


learning_result <- cram_learning(X, D, Y, batch, model_type = model_type,
                                 learner_type = learner_type, baseline_policy = baseline_policy,
                                 parallelize_batch = parallelize_batch, model_params = model_params,
                                 custom_fit = custom_fit, custom_predict = custom_predict)



print(learning_result)


# --------------------------------------------------------------------------------------

# Example usage of CRAM EXPERIMENT
set.seed(123)

# Custom X-Learner: Returns only the final model
custom_fit <- function(X, Y, D) {

  # Split the data into treated and control groups
  treated_indices <- which(D == 1)
  control_indices <- which(D == 0)

  X_treated <- X[treated_indices, ]
  Y_treated <- Y[treated_indices]
  X_control <- X[control_indices, ]
  Y_control <- Y[control_indices]

  # Step 1: Fit base models on treated and control groups separately
  model_treated <- cv.glmnet(as.matrix(X_treated), Y_treated, alpha = 0)
  model_control <- cv.glmnet(as.matrix(X_control), Y_control, alpha = 0)

  # Step 2: Compute pseudo-outcomes
  # Predict outcomes for the treated group using the control model
  tau_control <- Y_treated - predict(model_control, as.matrix(X_treated), s = "lambda.min")

  # Predict outcomes for the control group using the treated model
  tau_treated <- predict(model_treated, as.matrix(X_control), s = "lambda.min") - Y_control

  # Step 3: Combine pseudo-outcomes into a single training set
  X_combined <- rbind(X_treated, X_control)
  tau_combined <- c(tau_control, tau_treated)
  weights <- c(rep(1, length(tau_control)), rep(1, length(tau_treated))) # Equal weighting

  # Step 4: Fit a single model on the combined pseudo-outcomes
  final_model <- cv.glmnet(as.matrix(X_combined), tau_combined, alpha = 0, weights = weights)

  return(final_model)
}

# Custom prediction function
custom_predict <- function(model, X_new, D_new) {
  # Use the final model for predictions on new data
  cate <- predict(model, as.matrix(X_new), s = "lambda.min")
  as.numeric(cate) # Return as a numeric vector
}

## Generate data
n <- 10000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y

## Parameters
batch <- 20
model_type <- NULL # causal_forest, s_learner, m_learner
learner_type <- NULL # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL

## Run cram_experiment
experiment_results <- cram_experiment(X, D, Y, batch, model_type = model_type,
         learner_type = learner_type, baseline_policy = baseline_policy,
         parallelize_batch = parallelize_batch, model_params = model_params,
         custom_fit = custom_fit, custom_predict = custom_predict, alpha=alpha)


print(experiment_results)

# --------------------------------------------------------------------------------------
# Example usage of CRAM SIMULATION
set.seed(123)

## Obtain reference dataset X and define the data generation processes for D and Y
n <- 10000
sample_size <- 1000
data <- generate_data(n)
X <- data$X

dgp_X <- function(n) {
  data.table(
    binary = rbinom(n, 1, 0.5),
    discrete = sample(1:5, n, replace = TRUE),
    continuous = rnorm(n)
  )
}

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
nb_simulations <- 10
nb_simulations_truth <- 2  # nb_simulations_truth must be greater than nb_simulations
model_type <- "causal_forest" # causal_forest, s_learner, m_learner
learner_type <- NULL # NULL, ridge, fnn
alpha <- 0.05
baseline_policy <- as.list(rep(0, sample_size)) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL
custom_fit <- NULL
custom_predict <- NULL

## Run cram_experiment
# install.packages("profvis")
# library(profvis)
print(Sys.time())
simulation_results <- cram_simulation(X = NULL, dgp_X = dgp_X, dgp_D = dgp_D,
                                      dgp_Y, batch, nb_simulations, nb_simulations_truth, sample_size,
                                      model_type = model_type, learner_type = learner_type,
                                      alpha=0.05, baseline_policy = baseline_policy,
                                      parallelize_batch = parallelize_batch, model_params = model_params,
                                      custom_fit = custom_fit, custom_predict = custom_predict)

print(Sys.time())
print(simulation_results)

# --------------------------------------------------------------------------------------

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
parallelize_batch <- FALSE
model_params <- NULL
num_permutations <- 10

# Run Averaged CRAM
avg_cram_results <- averaged_cram(
  X = X, D = D, Y = Y, batch = batch,
  model_type = model_type, learner_type = learner_type,
  alpha = alpha, baseline_policy = baseline_policy,
  parallelize_batch = parallelize_batch, model_params = model_params,
  custom_fit = NULL, custom_predict = NULL,
  num_permutations = num_permutations
)

# Print Results
print(avg_cram_results$avg_policy_value)
print(avg_cram_results$var_policy_value)






