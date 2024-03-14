\name{deepGLM}
\alias{deepGLM}
\alias{deepGlm}
\alias{predict.deepGlm}
\alias{summary.deepGlm}
\alias{residuals.deepGlm}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Deep learning for the generalized linear model
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Fit generalized linear models (Gaussian, Binomial and Poisson) using deep learning neural network (DNN). The glm formula is specified by giving a symbolic description of the predictor and a description of the error distribution.
}
\usage{
deepGlm(formula, model, family = c("gaussian", "binomial",
        "poisson"), data, epochs = 200, lr_rate = 1e-04,
         batch_size = 64, alpha = 0.7, lambda = 1, verbose = 0,
         weights = NULL, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a formula expression as for other regression models. 
    The response is usually an object for glm response variable. See the documentation for 'glm', 'lm' and 'formula' for details.}
  \item{model}{a deep neural network model, created by function dNNmodel().}
  \item{family}{a description of the error distribution and link function to
       be used in the model. This can be either a character
       string of 'gaussian', 'binomial', or 'poisson', naming a family function, 
       or result of a call to a family function (See 'family' for details of family functions).)}
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
See \code{\link{dNNmodel}} for details on how to specify a deep learning model. 

The following parameters in 'dnnControl' will be used to control the model fit process.

'epochs': number of deep learning epochs, default is 30.

%% 'v_split': validation split, a numerical value in [0, 1), for portion of samples used for validation, default is 0.

'verbose': verbose = 1 for print out verbose during the model fit, 0 for not print.

When the variance for covariance matrix X is too large, please use xbar = scale(x) to standardize X.
}
\value{
%%  ~Describe the value returned
An object of class "deepGlm" is returned. The deepGlm object contains the following list components:
%%  If it is a LIST, use
  \item{x}{Covariates for glm model}
  \item{y}{Object for glm model}
  \item{model}{dnn model}
  \item{predictor}{predictor score mu = f(x)} 
  \item{risk}{risk score = exp(predictor)}
}
\references{
%% ~put references to the literature/web site here ~
Chollet, F. and Allaire J. J. (2017). Deep learning with R. Manning.
}
\author{
Chen, B. E. 
}
\note{For glm models with Gaussian, Binomial and Poisson only}

%% ~Make other sections like Warning with \section{Warning }{....} ~
\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{deepAFT}}, \code{\link{dNNmodel}}, 
\code{\link{predict.deepGlm}},
\code{\link{print.deepSurv}}, \code{\link{glm}}
}
\examples{
## Example for deep learning for glm models
  set.seed(101)
### define model layers
  model = dNNmodel(units = c(4, 3, 1), activation = c("elu", "sigmoid", "sigmoid"), 
                   input_shape = 3)
  x = matrix(runif(15), nrow = 5, ncol = 3)
  y = exp(x[, 1] + rnorm(5))
  
  fit = deepGlm(y ~ x, model, family = "gaussian") 
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{dnn}
\keyword{deepGLM}
