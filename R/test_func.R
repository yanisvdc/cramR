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
      # Convert batch assignement vector/list into a list where each
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

