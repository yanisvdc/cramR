# Function to generate sample data with heterogeneous treatment effects: positive, neutral, and adverse
generate_data <- function(n) {
  # Create X with one binary, one discrete, and one continuous variable
  X <- data.frame(
    binary = rbinom(n, 1, 0.5),                # Binary variable (0 or 1)
    discrete = sample(1:5, n, replace = TRUE),  # Discrete variable (integer values from 1 to 5)
    continuous = rnorm(n)                       # Continuous variable (normal distribution)
  )
  
  # Convert X to matrix form for grf package compatibility
  X <- as.matrix(X)
  
  # Treatment generation
  D <- rbinom(n, 1, 0.5)  # Binary treatment with 50% probability
  
  # Outcome generation with heterogeneous treatment effect
  # Define treatment effect for each group
  # - Group 1: High benefit (positive effect)
  # - Group 2: Neutral effect (small effect, around zero)
  # - Group 3: High adverse effect (negative effect)
  
  treatment_effect <- ifelse(
    X[, "binary"] == 1 & X[, "discrete"] <= 2,    # Group 1: High benefit
    1,
    ifelse(X[, "binary"] == 0 & X[, "discrete"] >= 4,  # Group 3: High adverse effect
           -1,
           0.1)  # Group 2: Neutral effect (small positive or negative)
  )
  
  # Define the outcome Y
  # Outcome depends on the treatment effect and random noise
  Y <- D * (treatment_effect + rnorm(n, mean = 0, sd = 1)) + 
    (1 - D) * rnorm(n)  # Outcome influenced by treatment and noise for untreated
  
  # Return as a list
  return(list(X = X, D = D, Y = Y))
}
