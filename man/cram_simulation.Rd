% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cram_simulation.R
\name{cram_simulation}
\alias{cram_simulation}
\title{CRAM Simulation}
\usage{
cram_simulation(
  X = NULL,
  dgp_X = NULL,
  dgp_D,
  dgp_Y,
  batch,
  nb_simulations,
  nb_simulations_truth = NULL,
  sample_size,
  model_type = "causal_forest",
  learner_type = "ridge",
  alpha = 0.05,
  baseline_policy = NULL,
  parallelize_batch = FALSE,
  model_params = NULL,
  custom_fit = NULL,
  custom_predict = NULL,
  propensity = NULL
)
}
\arguments{
\item{X}{Optional. A matrix or data frame of covariates for each sample inducing
empirically the DGP for covariates.}

\item{dgp_X}{Optional. A function to generate covariate data for simulations.}

\item{dgp_D}{A vectorized function to generate binary treatment assignments for each
sample.}

\item{dgp_Y}{A vectorized function to generate the outcome variable for each sample
given the treatment and covariates.}

\item{batch}{Either an integer specifying the number of batches
(which will be created by random sampling) or a vector of length
equal to the sample size providing the batch assignment (index)
for each individual in the sample.}

\item{nb_simulations}{The number of simulations (Monte Carlo replicates) to run.}

\item{nb_simulations_truth}{Optional. The number of additional simmulations
(Monte Carlo replicates) beyond nb_simulations
to use when calculating the true policy value difference (delta)
and the true policy value (psi)}

\item{sample_size}{The number of samples in each simulation.}

\item{model_type}{The model type for policy learning. Options include \code{"causal_forest"},
\code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.}

\item{learner_type}{The learner type for the chosen model. Options include \code{"ridge"}
for Ridge Regression and \code{"fnn"} for Feedforward Neural Network.
Default is \code{"ridge"}.}

\item{alpha}{Significance level for confidence intervals. Default is 0.05 (95\% confidence).}

\item{baseline_policy}{A list providing the baseline policy (binary 0 or 1) for each sample.
If \code{NULL}, defaults to a list of zeros with the same length
as the number of rows in \code{X}.}

\item{parallelize_batch}{Logical. Whether to parallelize batch processing
(i.e. the cram method learns T policies,
with T the number of batches. They are learned in parallel
when parallelize_batch is TRUE vs. learned sequentially using
the efficient data.table structure when parallelize_batch is FALSE,
recommended for light weight training). Defaults to \code{FALSE}.}

\item{model_params}{A list of additional parameters to pass to the model,
which can be any parameter defined in the model reference package.
Defaults to \code{NULL}.}

\item{custom_fit}{A custom, user-defined, function that outputs a fitted model given training data
(allows flexibility). Defaults to \code{NULL}.}

\item{custom_predict}{A custom, user-defined, function for making predictions given a fitted model
and test data (allow flexibility). Defaults to \code{NULL}.}

\item{propensity}{The propensity score model}
}
\value{
A list containing:
\describe{
  \item{\code{avg_proportion_treated}}{The average proportion of treated individuals across simulations.}
  \item{\code{avg_delta_estimate}}{The average delta estimate across simulations.}
  \item{\code{avg_delta_standard_error}}{The average standard error of delta estimates.}
  \item{\code{delta_empirical_bias}}{The empirical bias of delta estimates.}
  \item{\code{delta_empirical_coverage}}{The empirical coverage of delta confidence intervals.}
  \item{\code{avg_policy_value_estimate}}{The average policy value estimate across simulations.}
  \item{\code{avg_policy_value_standard_error}}{The average standard error of policy value estimates.}
  \item{\code{policy_value_empirical_bias}}{The empirical bias of policy value estimates.}
  \item{\code{policy_value_empirical_coverage}}{The empirical coverage of policy value confidence intervals.}
}
}
\description{
This function performs the cram method (simultaneous learning and evaluation)
on simulation data, for which the data generation process (DGP) is known.
The data generation process for X can be given directly as a function or
induced by a provided dataset via row-wise bootstrapping.
Results are averaged across Monte Carlo replicates for the given DGP.
}
\examples{
# Define data generation process (DGP) functions
X_data <- data.table::data.table(
  binary = rbinom(100, 1, 0.5),                 # Binary variable (0 or 1)
  discrete = sample(1:5, 100, replace = TRUE),  # Discrete variable (1 to 5)
  continuous = rnorm(100)                       # Continuous variable
)
dgp_D <- function(X) rbinom(nrow(X), 1, 0.5)
dgp_Y <- function(D, X) { theta <- ifelse(
X[, binary] == 1 & X[, discrete] <= 2,  # Group 1: High benefit
1,
ifelse(X[, binary] == 0 & X[, discrete] >= 4,  # Group 3: High adverse effect
-1,
0.1)  # Group 2: Neutral effect
)
Y <- D * (theta + rnorm(length(D), mean = 0, sd = 1)) +
  (1 - D) * rnorm(length(D))  # Outcome for untreated

return(Y)
}

# Parameters:
nb_simulations <- 10
nb_simulations_truth <- 2
batch <- 5

# Perform CRAM simulation
result <- cram_simulation(X = X_data, dgp_D = dgp_D, dgp_Y = dgp_Y,
                          batch = batch, nb_simulations = nb_simulations,
                          nb_simulations_truth = nb_simulations_truth,
                          sample_size=50)

# Access results
result$avg_delta_estimate
result$delta_empirical_bias

}
\seealso{
\code{\link[grf]{causal_forest}}, \code{\link[glmnet]{cv.glmnet}}, \code{\link[keras]{keras_model_sequential}}
}
