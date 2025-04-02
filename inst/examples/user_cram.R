# Install devtools if not already installed
install.packages("devtools")
install.packages("remotes")


# remotes::install_github("yanisvdc/cramR", ref = "batch", force=TRUE)

# Install the cramR package from your GitHub repository
devtools::install_github("yanisvdc/cramR", force=TRUE)

# Load the package
library(cramR)

library(data.table)
library(glmnet)

# Load data generator
path <- "C:/Users/yanis/Documents/Documents/cramR_user"
source(file.path(path, "cram_generate_data.R"))


# SEED
set.seed(123)

# DATA GENERATION:
n <- 1000
data <- generate_data(n)
X <- data$X
D <- data$D
Y <- data$Y


# CRAM POLICY - package models ---------------------------------------------------------------

batch <- 20
model_type <- 'causal_forest' # causal_forest, s_learner, m_learner
learner_type <- NULL # NULL for causal_forest, ridge or fnn for s_learner and m_learner
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
parallelize_batch <- FALSE
model_params <- NULL # NULL for default, causal_forest: list(num.trees = 100), ridge: list(alpha = 1)
# fnn: 
# default_model_params <- list(
#   input_layer = list(units = 64, activation = 'relu', input_shape = input_shape),  # Define default input layer
#   layers = list(
#   list(units = 32, activation = 'relu')
#   ),
#   output_layer = list(units = 1, activation = 'linear'),
#   compile_args = list(optimizer = 'adam', loss = 'mse'),
#   fit_params = list(epochs = 5, batch_size = 32, verbose = 0)
#   )
alpha <- 0.05

experiment_results <- cram_policy(X, D, Y, batch, model_type = model_type,
                                      learner_type = learner_type, baseline_policy = baseline_policy,
                                      parallelize_batch = parallelize_batch, model_params = model_params,
                                      alpha=alpha)
print(experiment_results)

# CRAM POLICY - custom models ---------------------------------------------------------------

# Custom X-Learner
custom_fit <- function(X, Y, D, n_folds = 5) {  
  
  # Split the data into treated and control groups
  treated_indices <- which(D == 1)
  control_indices <- which(D == 0)
  
  X_treated <- X[treated_indices, ]
  Y_treated <- Y[treated_indices]
  X_control <- X[control_indices, ]
  Y_control <- Y[control_indices]
  
  # Step 1: Fit base models on treated and control groups separately
  model_treated <- cv.glmnet(as.matrix(X_treated), Y_treated, alpha = 0, nfolds = n_folds)  # Pass n_folds here
  model_control <- cv.glmnet(as.matrix(X_control), Y_control, alpha = 0, nfolds = n_folds)  # Pass n_folds here
  
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
  final_model <- cv.glmnet(as.matrix(X_combined), tau_combined, alpha = 0, weights = weights, nfolds = n_folds)  # Pass n_folds here
  
  return(final_model)
}

# Custom prediction function
custom_predict <- function(model, X_new, D_new) {
  # Use the final model for predictions on new data
  cate <- predict(model, as.matrix(X_new), s = "lambda.min")
  as.numeric(cate) # Return as a numeric vector
}



experiment_results <- cram_policy(X, D, Y, batch, model_type = NULL,
                                      learner_type = learner_type, baseline_policy = baseline_policy,
                                      parallelize_batch = parallelize_batch, model_params = model_params,
                                      custom_fit = custom_fit, custom_predict = custom_predict, alpha=alpha)
print(experiment_results)


# CRAM ML - package models ----------------------------------------------------------------

# Load necessary libraries
library(caret)

# Set seed for reproducibility
set.seed(42)

# Generate example dataset
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_data <- rnorm(100)  # Continuous target variable for regression
data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included


# Define caret parameters for simple linear regression (no cross-validation)
caret_params_lm <- list(
  method = "lm",
  trControl = trainControl(method = "none")
)

# Define the batch count (not used in this simple example)
nb_batch <- 5
# nb_batch <- rep(1:5, each = 20)


# Run ML learning function
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,  # Linear regression model
  batch = nb_batch,
  loss_name = 'mse',
  caret_params = caret_params_lm
)

print(result)

# NB: possible loss_name and caret_params: 
# full parameter list: https://topepo.github.io/caret/model-training-and-tuning.html#model-training-and-parameter-tuning

# loss_name can be: "mse" (Mean Squared Error), "rmse" (Root Mean Squared Error), 
# "mae" (Mean Absolute Error), "logloss" (Binary Log Loss), "accuracy" (Classification Accuracy),
# "euclidean_distance" (Squared Euclidean Distance for K-Means Clustering)

# caret_params can include:
# - method: Specifies the machine learning algorithm (e.g., "lm" for linear regression, 
#   "rf" for random forest, "xgbTree" for XGBoost, "svmLinear" for Support Vector Machines)
# - trControl: Defines the resampling method (e.g., trainControl(method = "cv", number = 5) for 5-fold CV, 
#   or trainControl(method = "none") for no resampling)
# - tuneGrid: A data frame specifying hyperparameters for tuning (e.g., expand.grid(mtry = c(2, 3, 4)) for Random Forest)
# - metric: Specifies the performance metric for model selection (e.g., "RMSE" for regression, "Accuracy" for classification)
# - preProcess: Preprocessing steps to apply (e.g., c("center", "scale") for normalization)
# - importance: Boolean flag for computing variable importance (default = FALSE, often TRUE for tree-based models)

# CRAM ML - custom fit, predict and loss ------------------------------------------------------

# Set seed for reproducibility
set.seed(42)

# Define custom fit function (train model)
custom_fit <- function(data) {
  # Manually define the formula
  model <- lm(Y ~ x1 + x2 + x3, data = data)
  return(model)
}

# Define custom predict function
custom_predict <- function(model, data) {
  predictors_only <- data[, setdiff(names(data), "Y"), drop = FALSE]  # Exclude target column
  predict(model, newdata = predictors_only)
}

# Define custom loss function (Mean Squared Error)
custom_loss <- function(predictions, data) {
  actuals <- data$Y
  mse_loss <- (predictions - actuals)^2
  return(mse_loss)
}

# Run ML learning function with custom model
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,  # Linear regression model
  batch = nb_batch,
  custom_fit = custom_fit,
  custom_predict = custom_predict,
  custom_loss = custom_loss
)

# Print results
print(result)




# CRAM BANDIT -------------------------------------------------------


# Set random seed for reproducibility
set.seed(42)

# Define parameters
T <- 100  # Number of timesteps
K <- 4    # Number of arms

# Simulate a 3D array `pi` of shape (T, T, K)
# - First dimension: Individuals (context Xj)
# - Second dimension: Time steps (pi_t)
# - Third dimension: Arms (depth)
pi <- array(runif(T * T * K, 0.1, 1), dim = c(T, T, K))

# Normalize probabilities so that each row sums to 1 across arms
for (t in 1:T) {
  for (j in 1:T) {
    pi[j, t, ] <- pi[j, t, ] / sum(pi[j, t, ])  
  }
}

# Simulate arm selections (randomly choosing an arm)
arm <- sample(1:K, T, replace = TRUE)

# Simulate rewards (assume normally distributed rewards)
reward <- rnorm(T, mean = 1, sd = 0.5)

# Run CRAM Bandit Evaluation
cram_results <- cram_bandit(pi, arm, reward)

print(cram_results)



# CRAM BANDIT SIMULATION ------------------------------------------------

horizon       <- 500L
simulations   <- 100L
k = 4
d= 3

# Reward beta parameters of linear model
list_betas <- cramR:::get_betas(simulations, d, k)

bandit        <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.3)
policy <- cramR:::BatchContextualEpsilonGreedyPolicy$new(epsilon=0.1, batch_size=5)
# policy <- cramR:::BatchLinUCBDisjointPolicyEpsilon$new(alpha=1.0, epsilon=0.1, batch_size=1)
# policy <- cramR:::BatchContextualLinTSPolicy$new(v = 0.1, batch_size=1)


sim <- cram_bandit_sim(horizon, simulations,
                            bandit, policy,
                            alpha=0.05, do_parallel = FALSE)

print(sim)
