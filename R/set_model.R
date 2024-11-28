#' Set the Model Based on Type and Learner
#'
#' This function maps the model type and learner type to the corresponding model function. For Feedforward Neural Networks (FNNs),
#' it provides flexibility to configure the input layer, hidden layers, output layer, and compilation arguments using the \code{model_params} argument.
#'
#' @param model_type A string specifying the model type. Supported options are \code{"causal_forest"}, \code{"s_learner"}, \code{"m_learner"}.
#' @param learner_type A string specifying the learner type. Supported options depend on \code{model_type}.
#'                     For example, \code{"ridge"} and \code{"fnn"} are supported learners for \code{"s_learner"}.
#' @param model_params A list of named parameters to configure the model. The structure of \code{model_params} varies depending on the \code{learner_type}.
#'                     For FNNs, the following structure is used:
#'                     \describe{
#'                       \item{\code{input_layer}}{A list defining the input layer. Must include:
#'                           \code{units}: Number of units in the input layer.
#'                           \code{activation}: Activation function for the input layer.
#'                       \item{\code{layers}}{A list of lists, where each sublist specifies a hidden layer with:
#'                           \code{units}: Number of units in the layer.
#'                           \code{activation}: Activation function for the layer.}
#'                       \item{\code{output_layer}}{A list defining the output layer. Must include:
#'                           \code{units}: Number of units in the output layer.
#'                           \code{activation}: Activation function for the output layer (e.g., \code{"linear"} or \code{"sigmoid"}).}
#'                       \item{\code{compile_args}}{A list of arguments for compiling the model. Must include:
#'                           \code{optimizer}: Optimizer for training (e.g., \code{"adam"} or \code{"sgd"}).
#'                           \code{loss}: Loss function (e.g., \code{"mse"} or \code{"binary_crossentropy"}).
#'                           \code{metrics}: Optional list of metrics for evaluation (e.g., \code{c("accuracy")}).}
#'                     }
#'                     For other learners (e.g., \code{"ridge"} or \code{"causal_forest"}), \code{model_params} can include relevant hyperparameters.
#' @return The instantiated model object or the corresponding model function.
#' @examples
#' # Example: Causal Forest with default parameters
#' set_model("causal_forest", NULL, model_params = list(num.trees = 100))
#'
#' # Example: Ridge regression for S-learner
#' set_model("s_learner", "ridge")
#'
#' # Example: FNN for S-learner with custom layers and compile settings
#' set_model("s_learner", "fnn", model_params = list(
#'   input_layer = list(units = 128, activation = "relu", input_shape = ncol(X) + 1),
#'   layers = list(
#'     list(units = 64, activation = "relu"),
#'     list(units = 32, activation = "tanh")
#'   ),
#'   output_layer = list(units = 1, activation = "sigmoid"),
#'   compile_args = list(optimizer = "sgd", loss = "binary_crossentropy", metrics = c("accuracy"))
#' ))
#'
#' @export
set_model <- function(model_type, learner_type, model_params = list()) {
  if (model_type == "causal_forest") {
    # For Causal Forest
    model <- grf::causal_forest
  } else if (learner_type == "ridge") {
    # For S-learner with Ridge Regression
    model <- glmnet::cv.glmnet
  } else if (learner_type == "fnn") {
    # Determine the input shape based on model_type
    input_shape <- if (model_type == "s_learner") ncol(X) + 1 else ncol(X)

    # Default model configuration
    default_model_params <- list(
      input_layer = list(units = 64, activation = 'relu', input_shape = input_shape),  # Define default input layer
      layers = list(
        list(units = 32, activation = 'relu')
      ),
      output_layer = list(units = 1, activation = 'linear'),
      compile_args = list(optimizer = 'adam', loss = 'mse')
    )

    # Merge user-provided parameters with defaults
    model_params <- modifyList(default_model_params, model_params)

    # Create the model
    model <- keras_model_sequential()

    # Add the input layer
    model %>% layer_dense(
      units = model_params$input_layer$units,
      activation = model_params$input_layer$activation,
      input_shape = input_layer
    )

    # Add hidden layers
    for (layer in model_params$layers) {
      model %>% layer_dense(
        units = layer$units,
        activation = layer$activation
      )
    }

    # Add the output layer
    model %>% layer_dense(
      units = model_params$output_layer$units,
      activation = model_params$output_layer$activation
    )

    # Compile the model
    compile_args <- model_params$compile_args
    model %>% compile(
      optimizer = compile_args$optimizer,
      loss = compile_args$loss
    )

  } else {
    stop("Unsupported model_type or learner_type.")
  }

  return(model)
}
