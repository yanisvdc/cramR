#' Validate or Set the Baseline Policy
#'
#' This function validates a provided baseline policy or sets a default baseline policy of zeros for all individuals.
#'
#' @param baseline_policy A list representing the baseline policy for each individual. If \code{NULL}, a default baseline
#'                        policy of zeros is created.
#' @param n An integer specifying the number of individuals in the population.
#' @return A validated or default baseline policy as a list of numeric values.
#' @examples
#' # Example: Default baseline policy
#' baseline_policy <- test_baseline_policy(NULL, n = 10)
#'
#' # Example: Valid baseline policy
#' valid_policy <- as.list(rep(1, 10))
#' baseline_policy <- test_baseline_policy(valid_policy, n = 10)
#'
#' # Example: Invalid baseline policy
#' \dontrun{
#' invalid_policy <- c(1, 0, 1, 0)
#' baseline_policy <- test_baseline_policy(invalid_policy, n = 10)
#' }
#' @export

test_baseline_policy <- function(baseline_policy, n) {
  # Validate or set default baseline policy
  if (is.null(baseline_policy)) {
    return(as.list(rep(0, n)))  # Default: list of zeros with the same length as rows in X
  } else {
    # Validate baseline_policy if provided
    if (!is.list(baseline_policy)) {
      stop("Error: baseline_policy must be a list.")
    }
    if (length(baseline_policy) != n) {
      stop("Error: baseline_policy length must match the number of observations in X.")
    }
    if (!all(sapply(baseline_policy, is.numeric))) {
      stop("Error: baseline_policy must contain numeric values only.")
    }
    return(baseline_policy)  # Return validated baseline_policy
  }
}


#' Validate or Generate Batch Assignments
#'
#' This function validates a provided batch assignment or generates random batch assignments for individuals.
#'
#' @param batch Either an integer specifying the number of batches or a vector/list of batch assignments for all individuals.
#' @param n An integer specifying the number of individuals in the population.
#' @return A list containing:
#'   \describe{
#'     \item{\code{batches}}{A list where each element contains the indices of individuals assigned to a specific batch.}
#'     \item{\code{nb_batch}}{The total number of batches.}
#'   }
#' @examples
#' # Example: Generate random batch assignments
#' result <- test_batch(3, n = 9)
#' print(result)
#'
#' # Example: Validate a batch assignment vector
#' batch_vector <- c(1, 1, 2, 2, 3, 3, 1, 2, 3)
#' result <- test_batch(batch_vector, n = 9)
#' print(result)
#'
#' # Example: Invalid batch assignment
#' \dontrun{
#' invalid_batch <- c(1, 1, 2)
#' result <- test_batch(invalid_batch, n = 9)
#' }
#' @export
test_batch <- function(batch, n) {
  if (is.numeric(batch) && length(batch) == 1) {
    # `batch` is an integer, interpret it as `nb_batch`
    nb_batch <- batch
    # Assign randomly a batch to each index
    # To do so, shuffle the indices and repeat 1:nb_batch sequence until we go
    # through all the indices
    indices <- sample(1:n)  # Randomly shuffle the indices without replacement
    group_labels <- rep(1:nb_batch, length.out = n)  # Repeat labels 1 to nb_batch, filling up to n elements
    # Split the first object according to the vector of factor attribution
    # Each component is the factor level (batch index), and is associated to the vector
    # of indices in this factor level (batch)
    batches <- split(indices, group_labels)  # Split indices into batches
    return(list(batches = batches, nb_batch = nb_batch))
  } else if (is.list(batch) || is.vector(batch)) {
    if (length(batch) == n) {
      # Validate that all elements are numeric
      if (!all(sapply(batch, is.numeric))) {
        stop("`batch` must be a vector or list of numeric values.")
      }
      # Convert batch assignment vector/list into a list where each
      # component (batch index) is associated to the list of individuals indices in it
      batch_assinement <- unlist(batch)  # Ensure it's a vector
      batches <- split(1:n, batch_assinement)
      nb_batch <- length(batches)
      return(list(batches = batches, nb_batch = nb_batch))
    } else {
      stop("`batch` must be a vector/list of length equal to the population size, or a list of vectors of indices.")
    }
  } else {
    stop("`batch` must be either an integer or a list/vector of batch assignement for all individuals")
  }
}



check_lengths <- function(n, ...) {
  # Capture all inputs as a named list
  inputs <- list(...)
  input_names <- sapply(substitute(list(...))[-1], deparse)

  # Iterate through inputs to check lengths
  for (i in seq_along(inputs)) {
    if (length(inputs[[i]]) != n) {
      stop(paste0("Length mismatch: '", input_names[i],
                  "' has length ", length(inputs[[i]]),
                  " but expected length is ", n, "."))
    }
  }
}


retrieve_and_validate_model <- function(model_type, learner_type, model_params, X, custom_fit, custom_predict) {
  if (!is.null(model_type)) {
    # Retrieve model and validate user-specified parameters
    if (!is.null(learner_type) && learner_type == "fnn") {
      model_params <- validate_params_fnn(model_type, learner_type, model_params, X)
      model <- set_model(model_type, learner_type, model_params)
    } else {
      model <- set_model(model_type, learner_type, model_params)
      model_params <- validate_params(model, model_type, learner_type, model_params)
    }
  } else {
    # Custom mode: ensure custom_fit and custom_predict are specified
    if (is.null(custom_fit) || is.null(custom_predict)) {
      stop("As model_type is NULL (custom mode), custom_fit and custom_predict must be specified")
    }
    model <- NULL  # No predefined model in custom mode
  }
  return(list(model = model, model_params = model_params))
}


export_cluster_variables <- function(cl, learner_type, model_type, model_params, custom_fit = NULL, custom_predict = NULL) {
  # Base variables to export
  varlist <- c("X", "D", "fit_model", "model_predict")

  if (!is.null(model_type)) {
    # Case 1: Model type is not NULL
    if (!is.null(learner_type) && learner_type == "fnn") {
      # Ensures that learner_type is not NULL before checking its value.
      # This prevents an error when attempting to compare NULL with "fnn".
      # For FNN, we need to set the model in each worker node
      # as the keras structure cannot be exported (it would become a serialized object)
      varlist <- c(varlist, "set_model", "model_type", "learner_type", "model_params")
    } else {
      # Add variables for standard models (learner_type is not "fnn")
      varlist <- c(varlist, "model_type", "learner_type", "model_params")
    }
  } else {
    # Outer else: Custom model case
    # Add variables for custom models
    varlist <- c(varlist, "custom_fit", "custom_predict")
  }

  # Export the variables to the cluster
  clusterExport(cl, varlist = varlist, envir = environment())
}



create_cumulative_data <- function(X, D, Y, batches, nb_batch) {
  # Step 3: Create a data.table for cumulative batches
  # Initialize an empty list to store cumulative data for each batch
  cumulative_data_list <- lapply(1:nb_batch, function(t) {
    # Combine indices for batches 1 through t
    cumulative_indices <- unlist(batches[1:t])

    # Subset X, D, Y using cumulative indices
    list(
      t = t,  # Add t as the index
      # cumulative_index = list(cumulative_indices),  # Optional: Store cumulative indices as a list in one row
      X_cumul = list(X[cumulative_indices, ]),
      D_cumul = list(D[cumulative_indices]),
      Y_cumul = list(Y[cumulative_indices])
    )
  })

  # Convert the list to a data.table
  cumulative_data_dt <- rbindlist(cumulative_data_list)

  # Explicitly return the cumulative data table
  return(cumulative_data_dt)
}

expected_outcome <- function(probs) {
  # Extract numeric values from column names (e.g., "class0" → 0)
  class_values <- as.numeric(gsub("[^0-9.-]", "", names(probs)))

  if (anyNA(class_values)) {
    stop("Column names must include numeric class labels (e.g., 'class0', 'class1').")
  }

  # Convert to matrix for row-wise multiplication
  prob_matrix <- as.matrix(probs)

  # Calculate expected value per row
  rowSums(sweep(prob_matrix, 2, class_values, FUN = "*"))
}

