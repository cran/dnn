\name{survfit}
\alias{survfit.dSurv}
\title{
Compute a Survival Curve from a deepAFT or a deepSurv Model 
}
\description{
Computes the predicted survival function of a previously fitted deepAFT or deepSurv model.
}
\usage{
## S3 method for class 'deepAFT' or 'deepSurv'
\method{survfit}{dSurv}(formula, se.fit=TRUE, conf.int=.95, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{formula}{a deepAFT or deepSurv fit object.}
  \item{se.fit}{a logical value indicating whether standard errors shall be computed. Default is TRUE}
  \item{conf.int}{the level for a two-sided confidence interval on the survival curve. Default is 0.95}
  \item{\dots}{other unused arguments.}
}
\details{
  survfit.dSurv is called to compuate baseline survival function S_T0(t) from the deepAFT model \code{\link{deepAFT}}, where T0 = T/exp(mu), or log(T) = log(T) - mu.
  
  For the deepSurv model  \code{\link{deepAFT}}, survfit.dSurv evaluates the Nelson-Aalen estimate of the baseline survival function.

  The default method, survfit has its own help page. Use methods("survfit") to get all the methods for the survfit generic.
}

\value{
%%  ~Describe the value returned
survfit.deepAFT returns a list of predicted baseline survival function, cumulative hazard function and residuals. 

  \item{surv}{Predicted baseline survival function for T0=T/exp(mu).}
  \item{cumhaz}{Baseline cumulative hazard function, -log(surv).}
  \item{hazard}{Baseline hazard function.} 
  \item{varhaz}{Variance of the baseline hazard.}
  \item{residuals}{Martingale residuals of the (deepAFT) model.} 
  \item{std.err}{Standard error for the cumulative hazard function, if se.fit = TRUE.}

See \code{\link{survfit}} for more detail about other output values such as upper, lower, conf.type. 
Confidence interval is based on log-transformation of survival function. 
}
\author{Bingshu E. Chen}
\seealso{
 The default method for survfit \code{\link{survfit}},
\code{\link{predict.dSurv}}
}
%\examples{}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ survfit }
