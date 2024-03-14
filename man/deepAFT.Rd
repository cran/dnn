\name{deepAFT}
\alias{deepAFT}
\alias{deepAFT.default}
\alias{deepAFT.formula}
\alias{deepAFT.ipcw}
\alias{deepAFT.trans}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Deep learning for the accelerated failure time (AFT) model 
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Fit a deep learning survival regression model. 
These are location-scale models for an arbitrary transform of the time variable; 
the most common cases use a log transformation, leading to accelerated failure time models.
}
\usage{
deepAFT(x, ...)

\method{deepAFT}{formula}(formula, model, data, control = list(...), method =
                 c("BuckleyJames", "ipcw", "transform"), ...)

\method{deepAFT}{default}(x, y, model, control, ...)

\method{deepAFT}{ipcw}(x, y, model, control, ...)
# use:
#   deepAFT.ipcw(x, y, model, control)
# or
#   class(x) = "ipcw"
#   deepAFT(x, y, model, control)
# 
\method{deepAFT}{trans}(x, y, model, control, ...)
# use:
#   class(x) = "transform"
#   deepAFT(x, y, model, control)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a formula expression as for other regression models. 
    The response is usually a survival object as returned by the
          'Surv' function.  See the documentation for 'Surv', 'lm' and
          'formula' for details.}
  \item{model}{deep neural network model, see below for details.}
  \item{data}{a data.frame in which to interpret the variables named in the formula.}
  \item{x}{Covariates for the AFT model}
  \item{y}{Surv object for the AFT model}
  \item{method}{methods to handle censoring data in deep AFT model fit, 'BuckleyJames' for Buckley and James method, 
          'ipcw' for inverse probability censoring weights method.
          'transform' for transformation based on book of Fan and Gijbels (1996, page 168)}
  \item{control}{a list of control values, in the format produced by
          'dnnControl'. The default value 'dnnControl()'}
  \item{\dots}{optional arguments}
%%  \item{epochs}{number of deep learning epochs, default is 30.}
%% \item{batch_size}{batch size, default is 128. 'NaN' may be generated if batch size is too 
%%     small and there is not event in a batch.}
%%  \item{lr_rate}{learning rate}
%%  \item{weights}{an optional vector of 'prior weights' to be used in the
%%          fitting process. Should be NULL or a numeric vector.}
%%  \item{alpha}{alpha rate for gradient descent}
%%  \item{lambda}{regularization term for dnn weights}
%%  \item{verbose}{verbose = 1 for print out verbose during the model fit, 0 for not print.}
%%  \item{epsilon}{}

}
\details{
%%  ~~ If necessary, more details than the description above ~~
See "Deep learning with R" for details on how to build a deep learning model. 

The following parameters in 'dnnControl' will be used to control the model fit process.

'epochs': number of deep learning epochs, default is 100.

'batch_size': batch size, default is 128. 'NaN' may be generated if batch size is too small and there is not event in a batch. 

'verbose': verbose = 1 for print out verbose during the model fit, 0 for not print.

'epsilon': epsilon for convergence check, default is epsilon = 0.001.

'max.iter': number of maximum iteration, default is max.iter = 100.

'censor.groups': a vector for censoring groups. A KM curve for censoring will be fit for each group. If a matrix is provided, then a Cox model will be used to predict the censoring probability.

When the variance for covariance matrix X is too large, please use xbar = apply(x, 2, stndx) to standardize X.
}
\value{
%%  ~Describe the value returned
An object of class "deepAFT" is returned. The deepAFT object contains the following list components:

%%  If it is a LIST, use
  \item{x}{Covariates for the AFT model}
  \item{y}{Survival object for the AFT model, y = Surv(time, event)}
  \item{model}{A fitted artificial neural network (ANN) model}
  \item{mean.ipt}{mean survival or censoring time}
  \item{predictor}{predictor score mu = f(x)} 
  \item{risk}{risk score = exp(predictor)}
  \item{method}{method for deepAFT fitting, either Buckley-James, IPCW or transformed model}
}
\references{
%% ~put references to the literature/web site here ~
Buckley, J. and James, I. (1979). Linear regression with cencored data. Biometrika, 66, page 429-436.

Norman, P. Li, W., Jiang, W. and Chen, B. E. (2024). DeepAFT: A nonparametric accelerated failure time model with artificial neural network. Manuscript submitted to Statistics in Medicine.

Chollet, F. and Allaire J. J. (2017). Deep learning with R. Manning.
}
\author{
Chen, B. E. and Norman P.
}
\note{For right censored survival time only}

%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{print.deepAFT}}, \code{\link{survreg}}, \code{\link{ibs.deepAFT}}
}
\examples{
## Example for deep learning model for AFT survival data
  set.seed(101)
### define model layers
  model = dNNmodel(units = c(4, 3, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  x = matrix(runif(15), nrow = 5, ncol = 3)
  time = exp(x[, 1])
  status = c(1, 0, 1, 1, 1)
  fit = deepAFT(Surv(time, status) ~ x, model)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{deepAFT}
\keyword{dnnControl}
\keyword{dNNmodel}% use one of  RShowDoc("KEYWORDS")
% __ONLY ONE__ keyword per line
