\name{deepSurv}
\alias{deepSurv}
\alias{deepSurv.default}
\alias{summary.deepSurv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Deep learning for the Cox proportional hazards model 
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Fit a survival regression model under the Cox proportional hazards assumption using 
deep learning neural network (DNN). 
}
\usage{
deepSurv(formula, model, data, epochs = 200, lr_rate = 1e-04,
        batch_size = 64, alpha = 0.7, lambda = 1, verbose = 0,
        weights = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a formula expression as for other regression models. 
    The response is usually a survival object as returned by the
          'Surv' function.  See the documentation for 'Surv', 'lm' and
          'formula' for details.}
  \item{model}{a deep neural network model, created by function dNNmodel().}
  \item{data}{a data.frame in which to interpret the variables named in the formula.}
%%  \item{x}{Covariates for Cox model}
%%  \item{y}{Surv object for Cox model}
%%  \item{control}{a list of control values, in the format produced by
%%          'dnnControl'. The default value 'dnnControl()'}
  \item{epochs}{number of deep learning epochs, default is 200.}
  \item{batch_size}{batch size, default is 64. 'NaN' may be generated if batch size is too
     small and there is not event in a batch.}
  \item{lr_rate}{learning rate for the gradient descent algorithm, default is lr_rate = 1e-04.}
  \item{weights}{an optional vector of 'prior weights' to be used in the
          fitting process. Should be NULL or a numeric vector.}
  \item{alpha}{momentum rate for the gradient descent method, alpha takes value in [0, 1), default is alpha = 0.70.}
  \item{lambda}{L2 regularization parameter for deep learning.} 
  \item{verbose}{verbose = 1 for print out verbose during the model fit, 0 for not print.}
  \item{\dots}{optional arguments}

%%  \item{validation_split}{}
%%  \item{verbose}{}
%%  \item{epsilon}{}
%%  \item{max.iter}{}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
See "Deep learning with R" for details on how to build a deep learning model. 

The following parameters in 'dnnControl' will be used to control the model fit process.

'epochs': number of deep learning epochs, default is 30.

%% 'v_split': validation split, a numerical value in [0, 1), for portion of samples used for validation, default is 0.

'verbose': verbose = 1 for print out verbose during the model fit, 0 for not print.

'epsilon': epsilon for convergence check, default is epsilon = 0.001.

'max.iter': number of maximum iteration, default is max.iter = 30.

When the variance for covariance matrix X is too large, please use xbar = scale(x) to standardize X.
}
\value{
%%  ~Describe the value returned
An object of class "deepSurv" is returned. The deepSurv object contains the following list components:
%%  If it is a LIST, use
  \item{x}{Covariates for Cox model}
  \item{y}{Surv object for Cox model}
  \item{model}{dnn model}
  \item{predictor}{predictor score mu = f(x)} 
  \item{risk}{risk score = exp(predictor)}
}
\references{
%% ~put references to the literature/web site here ~
Katzman JL, Shaham U, Cloninger A, Bates J, Jiang T, Kluger Y. DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology 2018; 18: 24.
}
\author{
Chen, B. E. wrote the R code using the partial likelihood cost function proposed by Katzman et al (2018). 
}
\note{For right censored survival time only}

%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{deepAFT}}, \code{\link{deepGlm}},
\code{\link{print.deepSurv}}, \code{\link{survreg}}
}
\examples{
## Example for deep learning proportional hazards survival model
  set.seed(101)
### define model layers
  model = dNNmodel(units = c(4, 3, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  x = matrix(runif(15), nrow = 5, ncol = 3)
  time = exp(x[, 1])
  status = c(1, 0, 1, 1, 1)
  fit = deepSurv(Surv(time, status) ~ x, model = model)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{dnn} % use one of  RShowDoc("KEYWORDS")
\keyword{deepSurv}% __ONLY ONE__ keyword per line
