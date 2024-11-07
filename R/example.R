# path <- "C:/Users/yanis/OneDrive/Documents/cramR/R"
#
# # Load functions
# source(file.path(path, "cram_generate_data.R"))
# source(file.path(path, "cram_experiment.R"))
# source(file.path(path, "cram_simulation.R"))
#
# source(file.path(path, "cram_learning.R"))
#
# source(file.path(path, "cram_estimate.R"))
# source(file.path(path, "cram_variance_estimator.R"))
# source(file.path(path, "cram_policy_value_estimator.R"))


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
model_type <- "M-learner" # Causal Forest, S-learner, M-learner
learner_type <- "ridge" # NULL, ridge, FNN
alpha <- 0.05
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))

## Run cram_experiment
experiment_results <- cram_experiment(X, D, Y, batch, model_type,
                                      learner_type, alpha, baseline_policy)
print(experiment_results)

# --------------------------------------------------------------------------------------------

# Example usage of CRAM SIMULATION
set.seed(123)

## Obtain reference dataset X and define the data generation processes for D and Y
## dgp_D (resp. dgp_Y) takes individual-level data Xi (resp. Xi and Di) as inputs
n <- 1000
data <- generate_data(n)
X <- data$X
dgp_D <- function(Xi) {
  return(rbinom(1, 1, 0.5))
}
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

## Parameters
batch <- 20
nb_simulations <- 2
nb_simulations_truth <- 4  # nb_simulations_truth must be greater than nb_simulations
model_type <- "Causal Forest" # "Causal Forest", "S-learner", "M-learner"
learner_type <- NULL # NULL, "ridge", "FNN"
baseline_policy <- as.list(rep(0, nrow(X))) # as.list(rep(0, nrow(X))), as.list(sample(c(0, 1), nrow(X), replace = TRUE))
alpha <- 0.05

## Run cram_experiment
simulation_results <- cram_simulation(X, dgp_D, dgp_Y, batch,
                          nb_simulations, nb_simulations_truth,
                          model_type, learner_type,
                          alpha, baseline_policy)
print(simulation_results)
