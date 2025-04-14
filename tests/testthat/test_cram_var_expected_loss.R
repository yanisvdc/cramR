library(testthat)

test_that("cram_var_expected_loss returns a numeric value and correct type", {
  set.seed(123)
  loss <- cbind(rep(0, 100), matrix(rnorm(100 * 5), nrow = 100))  # 6 columns, first is zero
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_var_expected_loss(loss, batch_indices)
  expect_type(result, "double")
  expect_length(result, 1)
  expect_true(is.finite(result))
})

test_that("cram_var_expected_loss handles constant loss correctly", {
  loss <- cbind(rep(0, 100), matrix(1, nrow = 100, ncol = 5))  # constant loss values
  batch_indices <- split(1:100, rep(1:5, each = 20))

  result <- cram_var_expected_loss(loss, batch_indices)
  expect_equal(result, 0)
})

test_that("cram_var_expected_loss throws error for non-matrix loss input", {
  loss <- as.data.frame(cbind(rep(0, 100), matrix(rnorm(100 * 5), nrow = 100)))
  batch_indices <- split(1:100, rep(1:5, each = 20))

  expect_error(cram_var_expected_loss(loss, batch_indices), "`loss` must be a matrix")
})

test_that("cram_var_expected_loss throws error for non-list batch_indices", {
  loss <- cbind(rep(0, 100), matrix(rnorm(100 * 5), nrow = 100))
  batch_indices <- matrix(1:100, nrow = 5)

  expect_error(cram_var_expected_loss(loss, batch_indices), "`batch_indices` must be a list")
})

test_that("cram_var_expected_loss scales correctly with batch count", {
  set.seed(42)
  loss1 <- cbind(rep(0, 60), matrix(rep(1:60, 3), ncol = 3))  # 4 columns total
  batch_indices_1 <- split(1:60, rep(1:3, each = 20))

  loss2 <- cbind(rep(0, 60), matrix(rep(1:60, 6), ncol = 6))  # 7 columns total
  batch_indices_2 <- split(1:60, rep(1:6, each = 10))

  var1 <- cram_var_expected_loss(loss1, batch_indices_1)
  var2 <- cram_var_expected_loss(loss2, batch_indices_2)

  expect_false(isTRUE(all.equal(var1, var2)))
})
