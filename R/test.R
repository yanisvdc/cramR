example_function <- function(x, y, z = NULL, w = 5, ...) {}
formal_args <- formals(example_function)
print(formal_args)
# $x
#
# $y
#
# $z
# NULL
# $w
# 5
# $...
#

positional_args <- names(formal_args)[
  sapply(formal_args, function(arg) identical(arg, quote(expr = ))) & names(formal_args) != "..."
]
print(positional_args)
# [1] "x" "y"
