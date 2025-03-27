#' Cramming Policy Evaluation for Multi-Armed Bandit
#'
#' This function implements the armed bandit policy evaluation formula for
#' estimating \eqn{\Delta(\pi_T; \pi_0)} as given in the user-provided formula.
#'
#' @param pi A 3-d array, for each row j, column t, depth a, gives pi_t(Xj, a)
#' @param reward A vector of rewards
#' @param arm A vector of arms chosen
#' @param batch A vector or integer. If a vector, gives the batch assignment for each context.
#' If an integer, interpreted as the batch size and contexts are assigned to a batch in the order of the dataset.
#' @return The estimated policy value difference \eqn{\Delta(\pi_T; \pi_0)}.
#' @export

cram_bandit_est <- function(pi, reward, arm, batch=1) {

  # batch is here the batch size or the vector of batch assignment.

  dims_result <- dim(pi)

  if (is.numeric(batch) && length(batch) == 1) {
    n <- dims_result[1]
    # `batch` is an integer, interpret it as `batch_size`
    batch_size <- batch  # Guaranteed to be an integer since n is divisible by B
    nb_batch <- n / batch_size

    # Assign batch indices in order without shuffling
    indices <- 1:n
    group_labels <- rep(1:nb_batch, each = batch_size)  # Assign first B to batch 1, etc.

    # Split indices into batches
    batches <- split(indices, group_labels)

  } else {

    batch_assinement <- unlist(batch)
    batches <- split(1:n, batch_assinement)
    nb_batch <- length(batches)
    batch_size <- length(batches[[1]])

  }


  if (length(dims_result) == 3) {
    # Extract relevant dimensions
    nb_arms <- dims_result[3]
    nb_timesteps <- dims_result[2]

    sample_size <- nb_timesteps * batch_size


    ## POLICY SLICED: remove the arm dimension as Xj is associated to Aj

    # pi:
    # for each row j, column t, depth a, gives pi_t(Xj, a)

    # We do not need the last column and the first two rows
    # We only need, for each row j, pi_t(Xj, Aj), where Aj is the arm chosen from context j
    # Aj is the jth element of the vector arm, and corresponds to a depth index

    # drop = False to maintain 3D structure

    # pi <- pi[-c(1,2), -ncol(pi), , drop = FALSE]
    pi <- pi[-(1:(2*batch_size)), -ncol(pi), , drop = FALSE]

    # depth_indices <- arm[3:nb_timesteps]
    depth_indices <- arm[(2*batch_size+1):sample_size]

    pi <- extract_2d_from_3d(pi, depth_indices)

  } else {

    pi <- pi[, colSums(is.na(pi)) == 0, drop = FALSE]

    dims_result <- dim(pi)

    # 2D case
    nb_timesteps <- dims_result[2]

    sample_size <- nb_timesteps * batch_size

    # Remove the first two rows and the last column
    # pi <- pi[-c(1,2), -ncol(pi), drop = FALSE]
    pi <- pi[-(1:(2 * batch_size)), -ncol(pi), drop = FALSE]

  }

  # pi is now a (T-2)B x (T-1) matrix


  ## POLICY DIFFERENCE

  pi_diff <- pi[, -1] - pi[, -ncol(pi)]

  # pi_diff is a (T-2)B x (T-2) matrix


  ## MULTIPLY by Rj / pi_j-1

  # Get diagonal elements from pi (i, i+1 positions)
  # pi_diag <- pi[cbind(1:(nrow(pi)), 2:(ncol(pi)))]  # Vectorized indexing
  row_indices <- 1:nrow(pi)  # All row indices
  col_indices <- rep(2:ncol(pi), each = batch_size)  # Repeats column indices B times

  pi_diag <- pi[cbind(row_indices, col_indices)]

  # Create multipliers using vectorized operations
  # multipliers <- (1 / pi_diag) * reward[3:length(reward)]
  multipliers <- (1 / pi_diag) * reward[(2*batch_size+1):length(reward)]


  # Apply row-wise multiplication using efficient matrix operation
  mult_pi_diff <- pi_diff * multipliers  # Works via R's recycling rules (most efficient)


  # EXTRA STEP when batch size is not 1: average contexts in each batch

  # Sample data: mat is (T-2)*B rows x (T-2) columns
  group <- rep(1:(nrow(mult_pi_diff) %/% batch_size), each = batch_size)
  mult_pi_diff <- rowsum(mult_pi_diff, group) / batch_size


  ## AVERAGE TRIANGLE INF COLUMN-WISE

  mult_pi_diff[upper.tri(mult_pi_diff, diag = FALSE)] <- NA

  deltas <- colMeans(mult_pi_diff, na.rm = TRUE, dims = 1)  # `dims=1` ensures row-wise efficiency

  ## SUM DELTAS

  sum_deltas <- sum(deltas)


  ## ADD V(pi_1)

  pi_first_col <- pi[, 1]
  pi_first_col <- pi_first_col * multipliers


  # add the term for j = 2, this is only the rewards for batch 2! The probabilities cancel out
  r2 <- reward[(batch_size+1):(2*batch_size)]

  pi_first_col <- c(pi_first_col, r2)

  # V(pi_1) is the average
  v_pi_1 <-  mean(pi_first_col)

  ## FINAL ESTIMATE

  estimate <- sum_deltas + v_pi_1

  return(estimate)
}
