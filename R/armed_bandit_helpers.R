extract_2d_from_3d <- function(array3d, depth_indices) {
  # Get array dimensions
  dims <- dim(array3d)
  nrow <- dims[1]  # Rows
  ncol <- dims[2]  # Columns

  # Ensure depth_indices length matches required rows
  if (length(depth_indices) != nrow) {
    stop("The arm selection vector should have same length as the first dimension of the policy array.")
  }

  # Vectorized index calculation
  i <- rep(1:nrow, each = ncol)  # Row indices
  j <- rep(1:ncol, times = nrow) # Column indices
  k <- rep(depth_indices, each = ncol)  # Depth indices

  # Calculate linear indices for efficient extraction
  linear_indices <- i + (j - 1) * nrow + (k - 1) * nrow * ncol

  # Create result matrix using vectorized indexing
  result_matrix <- matrix(array3d[linear_indices], nrow = nrow, ncol = ncol, byrow = TRUE)

  return(result_matrix)
}

# # Define parameters
# T <- 5   # Total time points
# K <- 2   # Number of depth layers
#
# # Create a 3D array (T-2 x T-1 x K)
# array3d <- array(1:((T-2)*(T-1)*K), dim = c(T-2, T-1, K))
#
# # Create arm vector (must contain indices between 1 and K)
# arm <- c(1, 2, 1, 2, 1)  # Example arm vector
#
# # Vectorized index calculation
# depth_indices <- arm[3:T]  # Select arm[i+2] values
#
# res <- extract_2d_from_3d(array3d, depth_indices)







# # Define new test parameters
# T_new <- 6   # New total time points
# K_new <- 3   # New number of depth layers
#
# # Create a new 3D array (T-2 x T-1 x K)
# array3d_new <- array(1:((T_new-2)*(T_new-1)*K_new), dim = c(T_new-2, T_new-1, K_new))
#
# # Create a new arm vector (must contain indices between 1 and K)
# arm_new <- c(2, 3, 1, 2, 3, 1)  # Example arm vector
#
# # Extract depth indices from arm
# depth_indices_new <- arm_new[3:T_new]  # Select arm[i+2] values
#
# # Apply function to new data
# result_new <- extract_2d_from_3d(array3d_new, depth_indices_new)
#
# # Output results
# cat("\nNew 3D Array:\n")
# print(array3d_new)
#
# cat("\nNew Arm Vector:\n")
# print(arm_new)
#
# cat("\nNew Depth Indices:\n")
# print(depth_indices_new)
#
# cat("\nExtracted 2D Matrix:\n")
# print(result_new)
#
