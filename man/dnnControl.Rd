\name{dnnControl}
\alias{dnnControl}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Auxiliary function for \code{\link{dnnFit}} dnnFit
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
dnnControl is an auxiliary function for \code{\link{dnnFit}}. Typically only used internally by the dnn package, may be used to construct a control argument for the deep learning neural network model to specify parameters such as a loss function. 
}
\usage{
  dnnControl(loss = c("mse", "cox", "bin", "log", "mae"), epochs = 300, 
	     batch_size = 64, verbose = 0, lr_rate = 0.0001,  
	     alpha = 0.5, lambda = 1.0, epsilon = 0.01, max.iter = 100, 
	     censor.group = NULL, weights = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{loss}{loss function for the neural network model, "mse" for mean square error (guassian glm model), 
	  "mae" for mean absolute error, 
	  "cox" for the Cox partial likelihood (proportional hazards model), 
	  "bin" for cross-entropy (binomial glm model), 
	  "log" for log-linear (poisson glm model).}
  \item{epochs}{number of deep learning epochs, default is 30.}

  \item{batch_size}{batch size, default is 64. 'NaN' may be generated if batch size is too small and there is not event in a batch.} 
  \item{lr_rate}{learning rate, default is 0.0001.}
  \item{weights}{an optional vector of 'prior weights' to be used in the
          fitting process. Should be NULL or a numeric vector, default is NULL.}
  \item{alpha}{alpha decay rate for momentum gradient descent, default is 0.5.}
  \item{lambda}{regularization term for dnn weighting parameters, 0.5*lambda*W*W), default is 1.0.}
  \item{verbose}{verbose = 1 for print out verbose during the model fit, 0 for not print.}
  \item{epsilon}{epsilon for convergence check, default is epsilon = 0.01.}
  \item{max.iter}{number of maximum iteration, default is max.iter = 100. This is used in the deepAFT function}
  \item{censor.group}{a vector for censoring groups. A KM curve for censoring will be fit for each group. If a matrix is provided, then a Cox model will be used to predict the censoring probability. Used only in the deepAFT function.}
%%  \item{\dots}{optional arguments}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
dnnControl is used in model fitting of "dnnFit". Additional loss functions will be added to the library in the future. 
}
\value{
%%  ~Describe the value returned
This function checks the internal consistency and returns a list of values as input to control model fitting of "dnnFit".

%%  If it is a LIST, use
  %%\item{cost}{cost at the final epoch.}
}
\references{
%% ~put references to the literature/web site here ~

Norman, P. and Chen, B. E. (2023). DeepAFAT: A nonparametric accelerated failure time model with artificial neural network. Manuscript to be submitted. 
}
\author{
Chen, B. E.
}
\note{For right censored survival time only}

%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{deepAFT}}, \code{\link{deepGLM}}, \code{\link{deepSurv}}, \code{\link{dnnFit}}
}
\examples{
## Example for dnnControl
##
# model = dNNmodel()

  control = dnnControl(loss='mse')
  
# can also be used in   
# fit = dnnFit(y ~ x, model, control) 
# print(fit)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{dnnFit}% __ONLY ONE__ keyword per line
