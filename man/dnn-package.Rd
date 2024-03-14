\name{dnn-package}
\alias{dnn-package}
\alias{dnn-doc}
\alias{dnn}
\docType{package}
\title{
An R package for the deep neural networks probability and statistics models 
}
\description{
This package provides tools for deep neural network which allow user define loss function for complex outcome data with probability and statistics models such as generalized linear models, accelerated failure time (AFT) models, and Cox proportional hazards models.

It contains the essential building blocks such as feed forward network and back propagation. This gives users the flexibility to write their own loss function (i.e. cost function) and train the neural network.
}
\details{
\{dnn\} is a R package for deep learning neural network with probability models that use the negative of the log-likelihood as the loss function. It provides functions for feed forward network from covariates to the output layer and back propagation to find the derivatives of the weight parameters. 
Different optimization methods such as stochastic gradient descent (SGD), Momentum and ADAM can be used to train the network.

Currently, \{ dnn \} can be install by 

the package source file 'dnn.tar.gz', use

    install.packages("dnn.tar.gz", repos = NULL, type = "source")

users can use the following steps to install the most recent version of 'dnn' package:

1. First, you need to install the 'devtools' package. 
   You can skip this step if you have 'devtools' installed in your R. Invoke R and then type

   install.packages("devtools")

2. Load the devtools package.

   library(devtools)

3. Install "dnn" package from github with R command

  install_github("statapps/dnn")

A stable version of View the "dnn" package is also available from the Comprehensive R Archive Network 
(https://CRAN.R-project.org/package=dnn) and can be installed using R command 

  install.packages("dnn")

}
\author{Bingshu E. Chen

Maintainer: Bingshu E. Chen <bingshu.chen@queensu.ca>
}
%\references{}

\keyword{Deep Neural Networks}

\seealso{
\code{\link{dNNmodel}},  
\code{\link{bwdNN}},
\code{\link{fwdNN}},
\code{\link{deepAFT}}, 
\code{\link{deepGLM}}, 
\code{\link{deepSurv}}, 
\code{\link{coxph}}, 
\code{\link{glm}}
\code{\link{survival}}
}
\examples{
# Create the models with 3 layers
  model = dNNmodel(units=c(8, 6, 1), activation = c('elu', 'relu', 'sigmoid'), 
          input_shape = c(3))
  print(model)
# 
# Feed forward network with dummy data x
  x = matrix(runif(15), nrow = 5, ncol = 3)
  cache = fwdNN(x, model)
#
# Back propagation with dummy dy = dL/dyhat and minin batch for SGD
  dy = as.matrix(runif(5, -0.1, 0.1), nrow = 5)
  dW = bwdNN(dy, cache, model)
#
# Gradient descent with SGD 
  lr_rate = 0.0001
  sgd = function(w, dw) {w-lr_rate*dw}
  model$params = mapply(sgd, w = model$params, dw = dW)
}
