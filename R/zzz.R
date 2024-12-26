.onAttach <- function(libname, pkgname) {
  packageStartupMessage("cramR requires the data.table package. Ensure it is installed and loaded if needed.")
}
