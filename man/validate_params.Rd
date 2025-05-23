% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/validate_params.R
\name{validate_params}
\alias{validate_params}
\title{Cram Policy: Validate User-Provided Parameters for a Model}
\usage{
validate_params(model_function, model_type, learner_type, user_params)
}
\arguments{
\item{model_function}{The model function for which parameters are being validated (e.g., \code{grf::causal_forest}).}

\item{model_type}{The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}. Note: you can also set model_type to NULL and specify custom_fit and custom_predict to use your custom model.}

\item{learner_type}{The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression, \code{"fnn"} for Feedforward Neural Network and \code{"caret"} for Caret. Default is \code{"ridge"}. if model_type is 'causal_forest', choose NULL, if model_type is 's_learner' or 'm_learner', choose between 'ridge', 'fnn' and 'caret'.}

\item{user_params}{A named list of parameters provided by the user.}
}
\value{
A named list of validated parameters that are safe to pass to the model function.
}
\description{
This function validates user-provided parameters against the formal arguments of a specified model function.
It ensures that all user-specified parameters are recognized by the model and raises an error for invalid parameters.
}
