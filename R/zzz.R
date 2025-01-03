.onLoad <- function(libname, pkgname) {
  # Check if data.table is installed
  if (!requireNamespace("data.table", quietly = TRUE)) {
    # Attempt to install data.table
    install.packages("data.table")
  }

  # Load data.table silently
  requireNamespace("data.table", quietly = TRUE)
}
