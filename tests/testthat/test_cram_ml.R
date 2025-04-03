library(testthat)


test_that("Mean Squared Error (MSE) works", {
  y_true <- c(1, 2, 3)
  y_pred <- c(1.5, 2.5, 3.5)
  expected <- (y_pred - y_true)^2
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "se")
  expect_equal(actual, expected)
})

test_that("Mean Absolute Error (MAE) works", {
  y_true <- c(1, 2, 3)
  y_pred <- c(1.5, 2.5, 3.5)
  expected <- abs(y_pred - y_true)
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "ae")
  expect_equal(actual, expected)
})

test_that("Binary Log Loss works", {
  y_true <- c(0, 1, 0)
  y_pred <- c(0.1, 0.9, 0.2)
  epsilon <- 1e-15
  y_pred_clipped <- pmax(pmin(y_pred, 1 - epsilon), epsilon)
  expected <- - (y_true * log(y_pred_clipped) + (1 - y_true) * log(1 - y_pred_clipped))
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "logloss")
  expect_equal(actual, expected, tolerance = 1e-7)
})

test_that("Log Loss handles extreme probabilities", {
  y_true <- c(1, 0)
  y_pred <- c(1.0, 0.0)
  loss <- get_loss_function("logloss")
  clipped_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  expected <- c(-log(clipped_pred[1]), -log(1 - clipped_pred[2]))
  actual <- loss(y_pred, y_true)
  expect_equal(actual, expected)
})

test_that("Accuracy works", {
  y_true <- factor(c("A", "B", "A"))
  y_pred <- factor(c("A", "B", "B"))
  expected <- as.numeric(y_pred == y_true)
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "accuracy")
  expect_equal(actual, expected)
})

test_that("Euclidean Distance (K-Means) works", {
  data <- data.frame(x = c(1, 2, 5), y = c(1, 2, 5))
  ml_preds <- c(1, 1, 2)
  centroids <- aggregate(. ~ cluster, data = cbind(cluster = ml_preds, data), FUN = mean)
  assigned_centroids <- centroids[ml_preds, -1]
  expected <- rowSums((data - assigned_centroids)^2)
  actual <- compute_loss(ml_preds, data, loss_name = "euclidean_distance")
  expect_equal(actual, expected)
})

test_that("Unrecognized loss function throws error", {
  expect_error(get_loss_function("invalid_loss"), "not recognized")
})

test_that("Supervised loss with length mismatch errors", {
  y_true <- c(1, 2)
  y_pred <- c(1, 2, 3)
  expect_error(
    compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "se"),
    "same length"
  )
})

test_that("K-Means with incorrect ml_preds length errors", {
  data <- data.frame(x = 1:3, y = 4:6)
  ml_preds <- c(1, 2)
  expect_error(
    compute_loss(ml_preds, data, loss_name = "euclidean_distance"),
    "must be a vector"
  )
})

test_that("Accuracy with all correct predictions", {
  y_true <- factor(c("A", "A", "A"))
  y_pred <- factor(c("A", "A", "A"))
  expected <- rep(1, 3)
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "accuracy")
  expect_equal(actual, expected)
})

test_that("Accuracy with all incorrect predictions", {
  y_true <- factor(c("A", "B", "A"))
  y_pred <- factor(c("B", "A", "B"))
  expected <- rep(0, 3)
  actual <- compute_loss(y_pred, data.frame(y = y_true), formula = y ~ ., loss_name = "accuracy")
  expect_equal(actual, expected)
})
