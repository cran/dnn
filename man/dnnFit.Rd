\name{dnnFit}
\alias{dnnFit}
\alias{dnnFit2}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fitting a Deep Learning model with a given loss function
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
dnnFit is used to train a deep learning neural network model based on a specified loss function. 
}
\usage{
dnnFit(x, y, model, control)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{covariates for the neural network model}
  \item{y}{output (target) value for neural network model}
  \item{model}{the neural network model, see below for details}
  \item{control}{a list of control values, in the format produced by
          'dnnControl'. The default value is dnnControl(loss='mse')}
%%  \item{\dots}{optional arguments}
%%  \item{validation_split}{}
%%  \item{max.iter}{}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
The 'dnnFit' function takes the input data, the target values, the network architecture, and the loss function as arguments, and returns a trained model that minimizes the loss function. The function also supports various options for regularization and optimization of the model.

See \code{\link{dNNmodel}} for details on how to specify a deep learning model. 

Parameters in \code{\link{dnnControl}} will be used to control the model fit process. The loss function can be specified as dnnControl(loss = "lossFunction"). Currently, the following loss functions are supported: 

  'mse': Mean square error loss = 0.5*sum(dy^2)

  'cox': Cox partial likelihood loss = -sum(delta*(yhat - log(S0)))

  'bin': Cross-entropy = -sum(y*log(p) + (1-y)*log(1-p))

  'log': Log linear cost = -sum(y*log(lambda)-lambda)

  'mae': Mean absolute error loss = sum(abs(dy))

Additional loss functions will be added to the library in the future.

\{ dnnFit2 \} is a C++ version of dnnFit, which runs about 20\% faster, however, only loss = 'mse' and 'cox' are currently supported. 

When the variance for covariance matrix X is too large, please use xbar = scale(x) to standardize X.
}
\value{
%%  ~Describe the value returned
An object of class "dnnFit" is returned. The dnnFit object contains the following list components:

%%  If it is a LIST, use
  \item{cost}{cost at the final epoch.}
  \item{dW}{the gradient at the final epoch dW = dL/dW.}
  \item{fitted.values}{predictor value mu = f(x).}
  \item{history}{a cost history at each epoch.}
  \item{lp}{predictor value mu = f(x).}
  \item{logLik}{-2*log Likelihood = cost.}
  \item{model}{a dNNmodel object.}
  \item{residuals}{raw residual dy = d log(L)/dmu}
  \item{dvi}{deviance dvi = dy*dy}
}
\references{
%% ~put references to the literature/web site here ~
Buckley, J. and James, I. (1979). Linear regression with censored data. Biometrika, 66, page 429-436.

Norman, P. and Chen, B. E. (2019). DeepAFAT: A nonparametric accelerated failure time model with artificial neural network. Manuscript to be submitted. 

Chollet, F. and Allaire J. J. (2017). Deep learning with R. Manning.
}
\author{
Chen, B. E. and Norman P.
}
%\note{For right censored survival time only}

%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
  \code{\link{deepAFT}}, \code{\link{deepGlm}}, \code{\link{deepSurv}},   \code{\link{dnnControl}}

}
\examples{
## Example for dnnFit with MSE loss function to do a non-linear regression
  set.seed(101)
### define model layers
  model = dNNmodel(units = c(4, 3, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  x = matrix(runif(15), nrow = 5, ncol = 3)
  y = exp(x[, 1])
  control = dnnControl(loss='mse')
  fit = dnnFit(x, y, model, control) 
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{dnnFit}% __ONLY ONE__ keyword per line
