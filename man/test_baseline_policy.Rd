% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cram_helpers.R
\name{test_baseline_policy}
\alias{test_baseline_policy}
\title{Validate or Set the Baseline Policy}
\usage{
test_baseline_policy(baseline_policy, n)
}
\arguments{
\item{baseline_policy}{A list representing the baseline policy for each individual. If \code{NULL}, a default baseline
policy of zeros is created.}

\item{n}{An integer specifying the number of individuals in the population.}
}
\value{
A validated or default baseline policy as a list of numeric values.
}
\description{
This function validates a provided baseline policy or sets a default baseline policy of zeros for all individuals.
}
