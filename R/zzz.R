.onLoad <- function(libname, pkgname) {
  # Check if data.table is installed
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("The 'data.table' package is required but is not installed. Please install it using install.packages('data.table').")
  }

  # Load data.table silently
  requireNamespace("data.table", quietly = TRUE)
}
