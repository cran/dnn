\name{activation}
\alias{activation}
\alias{sigmoid}
\alias{elu}
\alias{relu}
\alias{lrelu}
\alias{idu}
\alias{dsigmoid}
\alias{delu}
\alias{drelu}
\alias{dlrelu}
\alias{dtanh}
\alias{didu}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Activation function
}
\description{
Different type of activation functions and the corresponding derivatives 
}
\usage{
  sigmoid(x)
  elu(x)
  relu(x)
  lrelu(x)
  idu(x)
  dsigmoid(y)
  delu(y)
  drelu(y)
  dlrelu(y)
  dtanh(y)   #activation function tanh(x) is already available in R
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{input of the activation function}
  \item{y}{input of the derivative of the activation function}
}

\details{
Each function returns either the activation function (e.g. sigmoid, relu) or its derivative (e.g. dsigmoid, drelu).
}

\value{
%%  ~Describe the value returned
An activation function is applied to x and returns a matrix the same size as x. 
The detail formula for each activation function is:
  \item{sigmoid}{return 1/(1+exp(-x))}
  \item{elu}{return x for x>0 and exp(x)-1 for x<0}
  \item{relu}{return x for x>0 and 0 for x<0}
  \item{lrelu}{return x for x>0 and 0.1*x for x<0}
  \item{tanh}{return tanh(x)}
  \item{idu}{return (x)}
}

\author{Bingshu E. Chen}
\seealso{
 \code{\link{bwdNN}}, 
 \code{\link{fwdNN}},
 \code{\link{dNNmodel}}, 
 \code{\link{optimizerSGD}},
 \code{\link{optimizerNAG}}
}
\examples{
  # Specify a dnn nodel with user define activation function in layer 2.
  softmax  = function(x) {log(1+exp(x))}    # y = log(1+exp(x))
  dsoftmax = function(y) {sigmoid(y)}       # x = exp(y)/(1+exp(y))
  model = dNNmodel(units=c(8, 6, 1), activation= c('relu', 'softmax', 'sigmoid'), 
          input_shape = c(3))
  print(model)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{activation function}
\keyword{Deep Neural Network}
\keyword{Feed forward}
\keyword{Back propagation}

