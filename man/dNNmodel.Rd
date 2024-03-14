\name{dNNmodel}
%- Also NEED an '\alias' for EACH other topic documented here.
\alias{dNNmodel}
%\alias{predict.dnn}

\title{
Specify a deep neural network model
}
\description{
\{dNNmodel\} is an R function to create a deep neural network model that is to be used 
in the feed forward network \{ fwdNN \} and back propagation \{ bwdNN \}.
}
\usage{
  dNNmodel(units, activation=NULL, input_shape = NULL, type = NULL, 
           N = NULL, Rcpp=TRUE, optimizer = c("momentum", "nag", "adam"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{units}{number of nodes for each layer}

  \item{activation}{activation function}

  \item{input_shape}{the number of columns of input X, default is NULL.} 

  \item{N}{the number of training sample, default is NULL.} 

  \item{type}{default is "dense", currently only support dense layer.} 

  \item{Rcpp}{use Rcpp (C++ for R) to speed up the fwdNN and bwdNN, default is "TRUE".} 

  \item{optimizer}{optimizer used in SGD, default is "momentum".} 

  %\item{\dots}{additional arguments to be passed to the low level regression fitting functions (see below).}  
}
\details{
dNNmodel returns an object of class "dNNmodel".

The function "print" (i.e., "print.dNNmodel") can be used to print a summary of the dnn model,

The function "summary" (i.e., "summary.dNNmodel") can be used to print a summary of the dnn model,
}
\value{
%%  ~Describe the value returned

An object of class "dNNmodel" is a list containing at least the following components:
  \item{units}{number of nodes for each layer}

  \item{activation}{activation function}

  \item{drvfun}{derivative of the activation function}
  
  \item{params}{the initial values of the parameters, to be updated in model training.} 

  \item{input_shape}{the number of columns of input X, default is NULL.}

  \item{N}{the number of training sample, default is NULL.}

  \item{type}{default is "dense", currently only support dense layer.} 
}
%\references{}
\author{
Bingshu E. Chen (bingshu.chen@queensu.ca)
}
%\note{}
%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{plot.dNNmodel}},
\code{\link{print.dNNmodel}},
\code{\link{summary.dNNmodel}},
\code{\link{fwdNN}},
\code{\link{bwdNN}},
\code{\link{optimizerSGD}},
\code{\link{optimizerNAG}},
}
\examples{
### To define a dnn model
 model = dNNmodel(units = c(8, 6, 1), activation = c("relu", "sigmoid", "sigmoid"), 
         input_shape = c(3))
}
\keyword{Deep Neural Network}
