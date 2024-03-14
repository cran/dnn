\name{fwdNN}
%- Also NEED an '\alias' for EACH other topic documented here.
\alias{fwdNN}
\alias{fwdNN2}
\alias{predict.dNNmodel}
\title{
  Feed forward and back propagation for dnn Models
}
\description{
\{fwdNN\} is an R function for feed forward network.
}
\usage{
   fwdNN(X, model)             
#
# to calculate a feed feedward model 
#
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{For "dNNmodel", X is a design matrix of dimension n * p.}

  \item{model}{a model return from dNNmodel function.}

  %\item{formula}{an object of class "formula"(or one that can be coerced to that class): a symbolic description of the model to be fitted. The details of model specification are given under 'Details'.}

  %\item{family}{a description of the response distribution and link function to be used in the model. 
%The available family function are either "binomial" for fitting a logistic regression model or "surv" for fitting a Cox proportional hazards model}

  %\item{data}{an optional data frame, list or environment (or object coercible by 'as.data.frame' to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which glm is called.}

  %\item{\dots}{additional arguments to be passed to the low level regression fitting functions (see below).}  
}
\details{
'cache' is the cache of each input layer, will be used in the bwdNN function.
}
\value{
%%  ~Describe the value returned
The function fwdNN return a list containing at least the following components:
  
  \item{cache}{a list contains the values of each output layer after activation function transformation and adding the 
intercept term (i.e. the bias term). The intercept does not add to the output layer in the cache.}
}
%\references{}
\author{
Bingshu E. Chen (bingshu.chen@queensu.ca)
}
%\note{}
%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{bwdNN}},
\code{\link{plot.dNNmodel}},
\code{\link{print.dNNmodel}},
\code{\link{summary.dNNmodel}},
}
\examples{
### define a dnn model, calculate the feed forward network
   model = dNNmodel(units = c(8, 6, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  
### feed forward with a dummy x matrix
   x = matrix(runif(15), nrow = 5, ncol = 3)
   cache = fwdNN(x, model)
}
\keyword{Deep Neural Network}
\keyword{Feed forward}