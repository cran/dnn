\name{hyperTuning}
\alias{hyperTuning}
\alias{CVpredErr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A function for tuning of the hyper parameters
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
\{ hyperTuning\} is a tuning tool to find the optimal hyper parameter for the ANN model.
}
\usage{
   hyperTuning(x, y, model, ER = c("cindex", "mse"), 
          method = c('BuckleyJames', 'ipcw', 'transform', 'deepSurv'), 
          lower = NULL, upper = NULL, node = FALSE,
          K = 5, R = 25)
### additional function used in hyperTuning is cross-validation prediction error
#
#  CVpredErr(x, y, model, control, method)
#
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{Covariates for the deep neural network model}
  \item{y}{Surv object for the deep neural network model}
  \item{model}{A deep neural network model, created by function dNNmodel().}
  \item{ER}{Prediction error measurement to be used in the cross vaditation, can be either a concordance index (cindex) or a mean square error (mse), default is cindex}
  \item{method}{Methods to handle censoring data in deep AFT model fit, 'BuckleyJames' for the Buckley and James method, 
      'ipcw' for the inverse probability censoring weights method.
      'transform' for the transformation method based on book of Fan and Gijbels (1996, page 168).
      'deepSurv' for the deepSurv model(Katzman, 2017)}
  \item{node}{Tuning the number of nodes in each hidden layer, default is FALSE}
  \item{K}{Number of folders of the cross-validatin, default is K = 5.}
  \item{lower, upper}{Bounds on the hyper parameters for the deep learning method. If NULL, then the default value for lower = dnnControl(alpha = 0.5, lambda = 1.0, lr_rate = 0.0001), upper = dnnControl(alpha = 0.97, lambda = 10, lr_rate = 0.001). }
  \item{R}{Number of random sample draw from the hyper parameter space, default is R = 25.}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
A random search method is used to optimal hyper parameter (Bergstra and Bengio, 2012). 
The function \{ CVpredErr\} will be call to calculate the cross-validation prediction error for the given x and y with the specified method from the input argument.
}
\value{
A list of "model" and "dnnControl" is returned.  The list contains at least the following components, 

  \item{model}{The "model" contains the optimal number of nodes for each hidden layer in the model specified by \code{\link{dNNmodel}}}
  \item{control}{The "control" contains the optimal tuning parameters with list components the same as those created by \code{\link{dnnControl}}}
}
\references{
Bergstra, J. and Bengio, Y. (2012). Random search for hyper-parameter optimization. The Journal of Machine Learning Research. 13, page 281-305.
}
\author{
Chen, B. E. (chenbe@queensu.ca)
}

%%\note{ }

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{deepAFT}}, \code{\link{deepGLM}}, \code{\link{deepSurv}}, \code{\link{dnnFit}}
}
\examples{
### Tuning the hyper parameter for a deepAFT model: 
#### cross-validation take a long time to run.
\donttest{
  set.seed(101)
### define model layers
  model = dNNmodel(units = c(4, 3, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  x = matrix(runif(45), nrow = 15, ncol = 3)
  time = exp(x[, 1])
  status = rbinom(15, 1, 0.5)
  y = Surv(time, status)
  ctl = dnnControl(epochs = 30)
  hyperTuning(x, y, model, method = "BuckleyJames", K = 2, R = 2, lower = ctl)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{ deepAFT }
\keyword{ deepSurv }
\keyword{ Hyper parameter }
