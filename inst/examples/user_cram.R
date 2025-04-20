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
path <- "C:/Users/yanis/Documents/Documents/cramR/inst/examples/"
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

# Options for batch:
# Either an integer specifying the number of batches or a vector/list of batch assignments for all individuals
batch <- 20

# Options for model_type: 'causal_forest', 's_learner', 'm_learner'
# Note: you can also set model_type to NULL and specify custom_fit and custom_predict to use your custom model
model_type <- 'causal_forest'

# Options for learner_type:
# if model_type == 'causal_forest', choose NULL
# if model_type == 's_learner' or 'm_learner', choose between 'ridge', 'fnn' and 'caret'
learner_type <- NULL

# Options for baseline_policy:
# A list representing the baseline policy assignment for each individual.
# If NULL, a default baseline policy of zeros is created.
# Example: all-zeros policy: as.list(rep(0, nrow(X))) / random policy:  as.list(sample(c(0, 1), nrow(X), replace = TRUE))
baseline_policy <- as.list(rep(0, nrow(X)))

# Whether to parallelize batch processing (i.e. the cram method learns T policies, with T the number of batches.
# They are learned in parallel when parallelize_batch is TRUE
# vs. learned sequentially using the efficient data.table structure when parallelize_batch is FALSE, recommended for light weight training).
# Defaults to FALSE.
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
# see vignettes for more details

alpha <- 0.05

experiment_results <- cram_policy(X, D, Y, batch, model_type = model_type,
                                      learner_type = learner_type, baseline_policy = baseline_policy,
                                      parallelize_batch = parallelize_batch, model_params = model_params,
                                      alpha=alpha)
print(experiment_results)

# CRAM POLICY - custom models ---------------------------------------------------------------

# Custom fit is called as follows internally:
# trained_model <- custom_fit(X, Y, D)
# It should be a function taking at least X, Y, D as parameters in this order.
# And returning a fitted model

# Custom predict is called as follows internally:
# learned_policy <- custom_predict(trained_model, X, D)
# It should be a function taking at least a fitted model, X, D as parameters in this order.
# And returning a vector of predictions (binary policy assignment) according to the fitted model (one prediction per row of the input data)

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
  as.numeric(cate > 0)  # 1 = treat, 0 = not treat
}


# Set model_type as NULL and specify custom_fit and custom_predict
experiment_results <- cram_policy(X, D, Y, batch, model_type = NULL,
                                      learner_type = learner_type, baseline_policy = baseline_policy,
                                      parallelize_batch = parallelize_batch, model_params = model_params,
                                      custom_fit = custom_fit, custom_predict = custom_predict, alpha=alpha)
print(experiment_results)


# If Y is categorical and you want to perform classification
# Cram policy works by outputting probabilities of class assignment
# Use caret classification models and do set classProbs = TRUE
# model_params contains an element formula to indicate the target variable and the predictors
# and it contains caret_params to indicate the caret::train parameters to use: https://topepo.github.io/caret/model-training-and-tuning.html

library(caret)

set.seed(43)
X <- matrix(rnorm(100 * 2), nrow = 100)
D <- sample(0:1, 100, replace = TRUE)
Y <- sample(c(0, 1), size = nrow(X), replace = TRUE)
batch <- rep(1:5, each = 20)
model_params <- list(formula = Y ~ ., caret_params = list(method = "rf", trControl = trainControl(method = "none", classProbs = TRUE)))

res <- cram_policy(
  X, D, Y, batch,
  model_type = "s_learner",
  learner_type = "caret",
  model_params = model_params
)

print(res)

# Note on M-learner:
# If model_type is m_learner, keep in mind that Y is transformed internally
# M-learner requires a propensity model and transformed outcomes
# propensity is an argument of cram_policy, if NULL, it defaults to propensity <- function(X) {rep(0.5, nrow(X))}
# outcome_transform is an element of model_params accessed as follows:
# outcome_transform <- model_params$m_learner_outcome_transform
# If NULL, it defaults to outcome_transform <- function(Y, D, prop_score) {Y * D / prop_score - Y * (1 - D) / (1 - prop_score)}, where prop_score is propensity(X)
# Thus, the Y transformed might not be categorical even though Y is categorical i.e. you might want to use a regression model


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

# Options for batch:
# Either an integer specifying the number of batches or a vector/list of batch assignments for all individuals
nb_batch <- 5


# Run ML learning function
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,  # Linear regression model
  batch = nb_batch,
  loss_name = 'se',
  caret_params = caret_params_lm
)

print(result)

# NB: possible loss_name and caret_params:
# full parameter list: https://topepo.github.io/caret/model-training-and-tuning.html#model-training-and-parameter-tuning

# loss_name can be: "se" (Squared Error), "ae" (Absolute Error), "logloss" (Log Loss), "accuracy" (Classification Accuracy),

# caret_params can include:
# - method: Specifies the machine learning algorithm (e.g., "lm" for linear regression,
#   "rf" for random forest, "xgbTree" for XGBoost, "svmLinear" for Support Vector Machines)
# - trControl: Defines the resampling method (e.g., trainControl(method = "cv", number = 5) for 5-fold CV,
#   or trainControl(method = "none") for no resampling)
# - tuneGrid: A data frame specifying hyperparameters for tuning (e.g., expand.grid(mtry = c(2, 3, 4)) for Random Forest)
# - metric: Specifies the performance metric for model selection (e.g., "RMSE" for regression, "Accuracy" for classification)
# - preProcess: Preprocessing steps to apply (e.g., c("center", "scale") for normalization)
# - importance: Boolean flag for computing variable importance (default = FALSE, often TRUE for tree-based models)

# Classification:  -------------------------------------------------------------

# Case 1: predict labels

# Use loss_name = accuracy (proportion of labels that match)
# Set classProbs to FALSE in trainControl
# Set classify = TRUE in cram_ml

set.seed(42)

# Generate example dataset
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_data <- rbinom(nrow(X_data), 1, 0.5)
data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

# classProbs is not specified and set to FALSE by default i.e. the model will output labels
caret_params_lm <- list(method = "rf", trControl = trainControl(method = "none"))


nb_batch <- 5

# Run ML learning function
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,
  batch = nb_batch,
  loss_name = 'accuracy',
  caret_params = caret_params_lm,
  classify = TRUE # indicate classification task
)

print(result)


# Case 2: predict probabilities

# Use loss_name = logloss
# Set classProbs to TRUE in trainControl
# Set classify = TRUE in cram_ml

# Set seed for reproducibility
set.seed(42)

# Generate example dataset
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_data <- rbinom(nrow(X_data), 1, 0.5)
data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

# Set classProbs = TRUE, the model will output probabilities of class assignment
caret_params_lm <- list(method = "rf", trControl = trainControl(method = "none", classProbs = TRUE))

nb_batch <- 5

# Run ML learning function
result <- cram_ml(
  data = data_df,
  formula = Y ~ .,
  batch = nb_batch,
  loss_name = 'logloss',
  caret_params = caret_params_lm,
  classify = TRUE # indicate classification task
)

print(result)


# CRAM ML - custom fit, predict and loss ------------------------------------------------------

# To use custom model and custom loss, simply do not specify loss_name and caret_params
# Instead, specify custom_fit, custom_predict and custom_loss

# Set seed for reproducibility
set.seed(42)

# Generate example dataset
X_data <- data.frame(x1 = rnorm(100), x2 = rnorm(100), x3 = rnorm(100))
Y_data <- rnorm(100)  # Continuous target variable for regression
data_df <- data.frame(X_data, Y = Y_data)  # Ensure target variable is included

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

# Define custom loss function (Squared Error)
custom_loss <- function(predictions, data) {
  actuals <- data$Y
  se_loss <- (predictions - actuals)^2
  return(se_loss)
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

# Consider the batched contextual linear armed bandit setting
# batch corresponds to the batch size here, let us note it B
# cram_bandit expects as inputs pi, arm, reward, batch (defaults to 1) and alpha (level of confidence interval)
# pi is an array giving for each context Xj, for each policy pi_t, for each arm a, the probability of arm selection pi_t(Xj, a)
# Thus, the natural shape of pi is (T*B, T, K), where T*B is the number of contexts, T is the number of policies, K is the number of arms
# We actually only use the probability of arm selection for the arm Aj that was selected under context Xj in the historical data
# Thus, the user can also provide a 2D array of shape (T*B, T)
# arm is the vector of arm chosen, of length T*B
# reward is the vector of rewards observed, of length T*B

# To calculate pi for your use case, please look at the armed_bandit_helpers file
# which contains functions to calculate these probabilities for the most common policies: Contextual Epsilon Greedy, UCB, Thompson Sampling


# Example with batch = 1

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
