# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/testing-design.html#sec-tests-files-overview
# * https://testthat.r-lib.org/articles/special-files.html

library(testthat)
library(cramR)

# Suppress the creation of __pycache__ directories during Python-related operations
Sys.setenv(PYTHONDONTWRITEBYTECODE = 1)

test_check("cramR")

testthat::teardown({
  tmp_dir <- tempdir()
  tf_temp_files <- list.files(tmp_dir, pattern = "^__autograph_generated_file.*\\.py$", full.names = TRUE)
  unlink(tf_temp_files)
})

