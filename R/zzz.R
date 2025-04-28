# R/zzz.R

.onLoad <- function(libname, pkgname) {
  # Set TensorFlow Autograph verbosity to suppress temp files
  Sys.setenv(TF_AUTOGRAPH_VERBOSITY = "0")
  options(quarto.enabled = FALSE)
  Sys.setenv(TF_AUTOGRAPH_VERBOSITY = "0")
  Sys.setenv(TF_DISABLE_AUTOGRAPH = "1")  # <- NEW LINE

  # # Optional: Set a custom temp directory (if needed)
  # Sys.setenv(TMPDIR = tempdir())
}

.onUnload <- function(libpath) {
  # Clean up autograph temp files safely
  temp_dir <- tempdir()
  autograph_files <- list.files(temp_dir, pattern = "^__autograph_generated_file.*\\.py$", full.names = TRUE)
  if (length(autograph_files) > 0) {
    try(unlink(autograph_files, recursive = TRUE, force = TRUE), silent = TRUE)
  }
}



