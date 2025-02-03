get_loss_function <- function(loss_name) {
  loss_functions <- list(

    # Mean Squared Error (MSE) - Per individual (Regression)
    # y_pred: Numeric vector of predictions
    # y_true: Numeric vector of true values
    "mse" = function(y_pred, y_true) {
      (y_pred - y_true) ^ 2
    },

    # Root Mean Squared Error (RMSE) - Per individual (Regression)
    "rmse" = function(y_pred, y_true) {
      sqrt((y_pred - y_true) ^ 2)
    },

    # Mean Absolute Error (MAE) - Per individual (Regression)
    "mae" = function(y_pred, y_true) {
      abs(y_pred - y_true)
    },

    # Binary Log Loss - Per individual (Classification)
    # y_pred: Numeric vector of predicted probabilities (values between 0 and 1)
    # y_true: Numeric vector of actual binary labels (0 or 1)
    "logloss" = function(y_pred, y_true) {
      y_pred <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)  # Prevent log(0) errors
      - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    },

    # Accuracy - Per individual (Classification)
    # y_pred: Factor or character vector of predicted class labels
    # y_true: Factor or character vector of actual class labels
    "accuracy" = function(y_pred, y_true) {
      as.numeric(y_pred == y_true)  # Returns 1 if correct, 0 if incorrect
    },

    # Squared Euclidean Distance - Per individual (Unsupervised K-Means)
    # points: Matrix where each row represents an individual's feature vector
    # centroids: Matrix where each row is the centroid assigned to the corresponding individual
    # Returns: Numeric vector of squared distances (one per individual)
    "euclidean_distance" = function(points, centroids) {
      rowSums((points - centroids) ^ 2)
    }
  )

  if (!(loss_name %in% names(loss_functions))) {
    stop("Error: Loss function not recognized. Choose from: ", paste(names(loss_functions), collapse = ", "))
  }

  return(loss_functions[[loss_name]])
}


compute_loss <- function(ml_preds, data, formula=NULL, loss_name) {

  # Get the appropriate loss function
  loss_function <- get_loss_function(loss_name)

  if (!is.null(formula)) {
    # Supervised Learning Case

    # Extract the response variable from the formula
    target_var <- all.vars(formula)[1]
    true_y <- data[[target_var]]  # Actual target values

    # Handle classification labels
    if (loss_name %in% c("logloss", "accuracy")) {
      # Convert factor to 0/1 for logloss
      if (loss_name == "logloss" && is.factor(true_y)) {
        if (nlevels(true_y) != 2) stop("logloss requires binary factor (0/1)")
        true_y <- as.numeric(true_y) - 1  # Convert to 0/1
      }
      # Ensure factors for accuracy
      if (loss_name == "accuracy" && !is.factor(true_y)) {
        true_y <- factor(true_y)
      }
    }

    # Ensure ml_preds and true_y have the same length
    if (length(ml_preds) != length(true_y)) {
      stop("Error: Predictions and true values must have the same length.")
    }

    # Compute per-instance loss
    individual_losses <- loss_function(y_pred = ml_preds, y_true = true_y)

  } else {
    # Unsupervised Learning Case

    if (loss_name == "euclidean_distance") {
      # K-Means Clustering: Distance to Centroid

      # Ensure ml_preds is a vector of cluster assignments
      if (!is.vector(ml_preds) || length(ml_preds) != nrow(data)) {
        stop("Error: `ml_preds` must be a vector of cluster assignments for K-Means, with length equal to the number of rows in `data`.")
      }

      # Compute centroids (one centroid per cluster)
      cluster_centers <- stats::aggregate(data, by = list(cluster = ml_preds), FUN = mean)[, -1]

      # Assign each individual its corresponding centroid
      assigned_centroids <- cluster_centers[ml_preds, ]

      # Compute per-instance squared Euclidean distance to assigned centroid
      individual_losses <- loss_function(points = data, centroids = assigned_centroids)

    } else {
      stop("Error: Unrecognized unsupervised loss function: ", loss_name)
    }
  }

  return(individual_losses)
}

# # install.packages(c("caret", "mlbench", "MASS"))
#
# library(caret)
# library(MASS)       # For Boston dataset
# library(mlbench)    # For PimaIndiansDiabetes2 dataset
#
# # --------------------------
# # 1. Regression Example (MSE, RMSE, MAE)
# # --------------------------
# data(Boston)
#
# # Train linear regression
# set.seed(123)
# lm_model <- train(medv ~ .,
#                   data = Boston,
#                   method = "lm",
#                   trControl = trainControl(method = "cv", number = 5))
#
# # Compute losses
# mse <- compute_loss(predict(lm_model, Boston), Boston, medv ~ ., "mse")
# cat("Regression MSE:", mean(mse), "(Caret RMSE^2:", lm_model$results$RMSE^2, ")\n")
#
# # --------------------------
# # 2. Classification Example (Log Loss, Accuracy)
# # --------------------------
# data(PimaIndiansDiabetes2)
# pima_data <- na.omit(PimaIndiansDiabetes2)
#
# # Convert target to factor with valid levels
# pima_data$diabetes <- factor(pima_data$diabetes, levels = c("neg", "pos"))
#
# # Train logistic regression
# set.seed(123)
# logreg_model <- train(diabetes ~ .,
#                       data = pima_data,
#                       method = "glm",
#                       family = "binomial",
#                       trControl = trainControl(method = "cv", number = 5))
#
# # Get predictions
# pred_probs <- predict(logreg_model, pima_data, type = "prob")$pos  # Probabilities
# pred_classes <- predict(logreg_model, pima_data)  # Class labels
#
# # Compute losses
# logloss <- compute_loss(pred_probs, pima_data, diabetes ~ ., "logloss")
# accuracy <- compute_loss(pred_classes, pima_data, diabetes ~ ., "accuracy")
#
# cat("Classification Results:\n",
#     "Mean Log Loss:", mean(logloss), "\n",
#     "Accuracy:", mean(accuracy), "\n")
#
# # --------------------------
# # 3. K-Means Example (Euclidean Distance)
# # --------------------------
# data(iris)
# features <- iris[, 1:4]
#
# # Train K-Means
# set.seed(123)
# kmeans_model <- kmeans(features, centers = 3)
#
# # Compute distances
# kmeans_loss <- compute_loss(kmeans_model$cluster, features, loss_name = "euclidean_distance")
# cat("K-Means Total Within-SS:", sum(kmeans_loss), "(Model's Within-SS:", kmeans_model$tot.withinss, ")\n")



create_cumulative_data_ml <- function(data, batches, nb_batch) {
  # Step 3: Create a data.table for cumulative batches
  # Initialize an empty list to store cumulative data for each batch
  cumulative_data_list <- lapply(1:nb_batch, function(t) {
    # Combine indices for batches 1 through t
    cumulative_indices <- unlist(batches[1:t])

    # Subset X, D, Y using cumulative indices
    list(
      t = t,  # Add t as the index
      # cumulative_index = list(cumulative_indices),  # Optional: Store cumulative indices as a list in one row
      data_cumul = list(data[cumulative_indices, ])
    )
  })

  # Convert the list to a data.table
  cumulative_data_dt <- rbindlist(cumulative_data_list)

  # Explicitly return the cumulative data table
  return(cumulative_data_dt)
}
