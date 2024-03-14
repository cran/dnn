\name{optimizerSGD}
\alias{optimizerSGD}
\alias{optimizerMomentum}
\alias{optimizerAdamG}
\alias{optimizerNAG}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Functions to optimize the gradient descent of a cost function
}
\description{
Different type of optimizer functions such as SGD, Momentum, AdamG and NAG.
}
\usage{
  optimizerMomentum(V, dW, W, alpha = 0.63, lr = 1e-4, lambda = 1) 
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{V}{Momentum V = alpha*V - lr*(dW + lambda*W); W = W + V. 
           NAG V = alpha*(V - lr*(dW + lambda*W); W = W + V - lr*(dW + lambda*W)}
  \item{dW}{derivative of cost with respect to W, can be founde by dW = bwdNN2(dy, cache, model), }
  \item{W}{weights for DNN model, optimizerd by W = W + V}
  \item{alpha}{Momentum rate 0 < alpha < 1, default is alpah = 0.5.}
  \item{lr}{learning rate, default is lr = 0.001.}
  \item{lambda}{regulation rate for cost + 0.5*lambda*||W||, default is lambda = 1.0.}
}

\details{
  For SGD with momentum, use
 
  V = 0; obj = optimizerMomentum(V, dW, W); V = obj$V; W = obj$W

  For SDG with MAG

  V = 0; obj = optimizerNAG(V, dW, W); V = obj$V; W = obj$W
   
}

\value{
%%  ~Describe the value returned
  return and updated W and other parameters such as V, V1 and V2 that will be used on SGD.
}

\author{Bingshu E. Chen}
\seealso{
 \code{\link{activation}}, 
 \code{\link{bwdNN}}, 
 \code{\link{fwdNN}},
 \code{\link{dNNmodel}}, 
 \code{\link{dnnFit}} 
}
%\examples{} 
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ optimizer SDG }
\keyword{ optimizer Momentum}
\keyword{ optimizer NAG}
\keyword{ optimizer AdamG}
