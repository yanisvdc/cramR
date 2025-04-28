library(testthat)
library(data.table)
library(contextual)

test_that("cram_bandit_sim runs and returns expected structure with  batch epsilon greedy", {

  horizon     <- 100L
  simulations <- 10L
  k <- 4
  d <- 3

  list_betas <- cramR:::get_betas(simulations, d, k)
  bandit <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.3)
  policy <- cramR:::BatchContextualEpsilonGreedyPolicy$new(epsilon = 0.1, batch_size = 5)

  result <- cram_bandit_sim(horizon, simulations, bandit, policy)

  expect_type(result, "list")
  expect_named(result, c("estimates", "summary_table", "interactive_table"))
})

test_that("cram_bandit_sim runs and returns expected structure with  batch = 1 epsilon greedy", {

  horizon     <- 100L
  simulations <- 10L
  k <- 4
  d <- 3

  list_betas <- cramR:::get_betas(simulations, d, k)
  bandit <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.3)
  policy <- cramR:::BatchContextualEpsilonGreedyPolicy$new(epsilon = 0.1, batch_size = 1)

  result <- cram_bandit_sim(horizon, simulations, bandit, policy)

  expect_type(result, "list")
  expect_named(result, c("estimates", "summary_table", "interactive_table"))
})

test_that("cram_bandit_sim runs and returns expected structure with ucb", {

  horizon     <- 100L
  simulations <- 10L
  k <- 4
  d <- 3

  list_betas <- cramR:::get_betas(simulations, d, k)
  bandit <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.3)
  policy <- cramR:::BatchLinUCBDisjointPolicyEpsilon$new(alpha=1.0, epsilon=0.1, batch_size=1)

  result <- cram_bandit_sim(horizon, simulations, bandit, policy)

  expect_type(result, "list")
  expect_named(result, c("estimates", "summary_table", "interactive_table"))
})


test_that("cram_bandit_sim runs and returns expected structure with thompson", {

  horizon     <- 100L
  simulations <- 10L
  k <- 4
  d <- 3

  list_betas <- cramR:::get_betas(simulations, d, k)
  bandit <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.3)
  policy <- cramR:::BatchContextualLinTSPolicy$new(v = 0.1, batch_size=1)

  result <- cram_bandit_sim(horizon, simulations, bandit, policy)

  expect_type(result, "list")
  expect_named(result, c("estimates", "summary_table", "interactive_table"))
})


test_that("cram_bandit_sim returns finite numeric outputs", {

  horizon     <- 20L
  simulations <- 5L
  k <- 2
  d <- 2

  list_betas <- cramR:::get_betas(simulations, d, k)
  bandit <- cramR:::ContextualLinearBandit$new(k = k, d = d, list_betas = list_betas, sigma = 0.2)
  policy <- cramR:::BatchContextualEpsilonGreedyPolicy$new(epsilon = 0.1, batch_size = 2)

  result <- cram_bandit_sim(horizon, simulations, bandit, policy)

  expect_true(is.finite(result$summary_table$Value[1]))
  expect_false(anyNA(result$estimates$estimate))
})
