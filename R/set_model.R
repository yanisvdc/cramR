#' Set Model
#'
#' This function maps the model type and learner type to the corresponding model function.
#'
#' @param model_type The model type for policy learning. Options include \code{"causal_forest"}, \code{"s_learner"}, and \code{"m_learner"}. Default is \code{"causal_forest"}.
#' @param learner_type The learner type for the chosen model. Options include \code{"ridge"} for Ridge Regression and \code{"fnn"} for Feedforward Neural Network. Default is \code{"ridge"}.
#' @param model_params A list of additional parameters to pass to the model, which can be any parameter defined in the model reference package. Defaults to \code{NULL}.
#'                     For FNNs, the following elements are defined in the model params list:
#'                     \describe{
#'                       \item{\code{input_layer}}{A list defining the input layer. Must include:
#'                         \describe{
#'                           \item{\code{units}}{Number of units in the input layer.}
#'                           \item{\code{activation}}{Activation function for the input layer.}
#'                           \item{\code{input_shape}}{Input shape for the layer.}
#'                         }
#'                       }
#'                       \item{\code{layers}}{A list of lists, where each sublist specifies a hidden layer with:
#'                         \describe{
#'                           \item{\code{units}}{Number of units in the layer.}
#'                           \item{\code{activation}}{Activation function for the layer.}
#'                         }
#'                       }
#'                       \item{\code{output_layer}}{A list defining the output layer. Must include:
#'                         \describe{
#'                           \item{\code{units}}{Number of units in the output layer.}
#'                           \item{\code{activation}}{Activation function for the output layer (e.g., \code{"linear"} or \code{"sigmoid"}).}
#'                         }
#'                       }
#'                       \item{\code{compile_args}}{A list of arguments for compiling the model. Must include:
#'                         \describe{
#'                           \item{\code{optimizer}}{Optimizer for training (e.g., \code{"adam"} or \code{"sgd"}).}
#'                           \item{\code{loss}}{Loss function (e.g., \code{"mse"} or \code{"binary_crossentropy"}).}
#'                           \item{\code{metrics}}{Optional list of metrics for evaluation (e.g., \code{c("accuracy")}).}
#'                         }
#'                       }
#'                     }
#'                     For other learners (e.g., \code{"ridge"} or \code{"causal_forest"}), \code{model_params} can include relevant hyperparameters.
#' @return The instantiated model object or the corresponding model function.
#' @examples
#' # Example: Causal Forest with default parameters
#' set_model("causal_forest", NULL, model_params = list(num.trees = 100))
#' @export

set_model <- function(model_type, learner_type, model_params) {
  if (model_type == "causal_forest") {
    # For Causal Forest
    model <- grf::causal_forest
    return(model)
  } else if (learner_type == "ridge") {
    # For Ridge Regression
    model <- glmnet::cv.glmnet
  } else if (learner_type == "caret") {
    # For Caret
    model <- caret::train
  } else if (learner_type == "fnn") {
    # For FNN using Keras
    model <- keras_model_sequential()

    # Add the input layer
    model %>% layer_dense(
      units = model_params$input_layer$units,
      activation = model_params$input_layer$activation,
      input_shape = model_params$input_layer$input_shape
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
