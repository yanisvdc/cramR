library(testthat)

test_that("cram_expected_loss computes expected result for valid input", {
  set.seed(123)

  # Simulate a loss matrix: 10 data points, 6 columns (1 zero + 5 batch losses)
  loss <- cbind(rep(0, 10), matrix(runif(10 * 5), nrow = 10))  # 10 x 6 matrix

  # Create batch indices for 5 batches
  batch_indices <- split(1:10, rep(1:5, each = 2))  # 5 batches of 2 points each

  result <- cram_expected_loss(loss, batch_indices)

  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})
