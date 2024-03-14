\name{bwdNN}
%- Also NEED an '\alias' for EACH other topic documented here.
\alias{bwdNN}
\alias{bwdNN2}
\alias{bwdCheck}
%\alias{dParamsCheck}
%\alias{getV}
\title{
  Back propagation for dnn Models
}
\description{
\{bwdNN\} is an R function for back propagation in DNN network.
}
\usage{
#
# To apply back propagation in with a feed forward model 
#
# use 
#
   bwdNN(dy, cache, model)
#
# to calculate derivative of dL/dW
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dy}{the derivative of the cost function with respect to the output layer of the fwdNN function.}
  \item{cache}{the cached output of fwdNN.}
  \item{model}{a model return from dNNmodel function.}
  %\item{\dots}{additional arguments to be passed to the low level regression fitting functions(see below).}  
  %\item{batch_size}{the size of the minin batch used for the stochastic gradient descent (SGD) optimization, the default value is 200.}

}
\details{
Here 'dy' plays an import role in the back propagation \{ bwdNN \} 
since the probability model's loss function takes the output 
layer of the \{ dnn \} (denote as yhat) as one of its parameter. 
Then 'dy' equals to the partial derivative of the loss function (-Log Likelihood) with respect to yhat, 
that is, dy = dL/d(yhat). 
For example, if the 'dnn' predicts the probability (yhat = p) for the mixture of two populations f1 and f2, 
then the likelihood function is f = p*f1 + (1-p)*f2, and 
the loss function is L = -log(p*f1+(1-p)*f2). Hence, dy = dL/dp = -(f1-f2)/f.

'cache' is the cache of each input layer generated from the \{ fwdNN \} function.

The function \{ bwdCheck \} calculates the numerical derivatives of dL/dW, which can be used to check if the back propagation is correct or not, see example below.
}
\value{
%%  ~Describe the value returned
A list contains the derivatives of weight parameter W is returned.
}
%\references{}
\author{
Bingshu E. Chen (bingshu.chen@queensu.ca)
}
%\note{}
%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{dNNmodel}},
\code{\link{fwdNN}},
\code{\link{plot.dNNmodel}},
\code{\link{print.dNNmodel}},
\code{\link{summary.dNNmodel}},
}
\examples{
### define a dnn model, calculate the feed forward network
   model = dNNmodel(units = c(8, 6, 1), 
           activation = c("elu", "sigmoid", "sigmoid"), input_shape = 3)
   print(model)
   x = matrix(runif(15), nrow = 5, ncol = 3)
   cache = fwdNN(x, model)
   # dy = dL/dp, where L is the cost function such as the 
   # log-likehood and p is the output layer parameter of the DNN
   dy = as.matrix(runif(5, -0.1, 0.1), nrow = 5)  # a dummy dy for bwdNN input
   y  = predict(model, x) + dy
   
   # back propagation 
   dW = bwdNN(dy, cache, model)
   dw = bwdCheck(x, y, model)
   print(dW[[1]])
   print(dw[[1]])
} 
\keyword{Deep Neural Network}
\keyword{Back propagation}
