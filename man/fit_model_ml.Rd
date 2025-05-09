% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit_model_ml.R
\name{fit_model_ml}
\alias{fit_model_ml}
\title{Cram ML: Fit Model ML}
\usage{
fit_model_ml(data, formula, caret_params, classify)
}
\arguments{
\item{data}{The dataset}

\item{formula}{The formula}

\item{caret_params}{The parameters for caret model}

\item{classify}{Indicate if this is a classification problem. Defaults to FALSE}
}
\value{
The fitted model object.
}
\description{
This function trains a given unfitted model with the provided data and parameters,
according to model type and learner type.
}
