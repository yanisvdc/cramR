% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_model.R
\name{fit_model}
\alias{fit_model}
\title{Fit a Provided Model Based on Type and Learner}
\usage{
fit_model(model, X, Y, W, model_type, learner_type, model_params)
}
\arguments{
\item{model}{An unfitted model object, as returned by `set_model`.}

\item{X}{A matrix or data frame of covariates for the samples.}

\item{Y}{A vector of outcome values.}

\item{W}{(Optional) A vector of binary treatment indicators (for models requiring treatment data, e.g., M-learner, Causal Forest).}

\item{model_type}{A string specifying the type of model. Supported options: "Causal Forest", "S-learner", "M-learner".}

\item{learner_type}{A string specifying the type of learner. Supported options: "ridge", "FNN".}

\item{model_params}{A list of additional parameters to pass to the fitting process.}
}
\value{
The fitted model object.
}
\description{
This function trains a given model with the provided data and parameters, taking into account the model type and learner type.
}
\examples{
# Example usage for Ridge Regression S-learner
model <- set_model("S-learner", "ridge")
fitted_model <- fit_model(model, X, Y, W = D, "S-learner", "ridge", model_params = list(alpha = 0))

# Example usage for FNN S-learner
model <- set_model("S-learner", "FNN")
fitted_model <- fit_model(model, X, Y, W = D, "S-learner", "FNN", model_params = list(epochs = 20))

}
