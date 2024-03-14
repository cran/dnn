\name{rsurv}
\alias{rsurv}
\alias{rSurv}
\alias{dsurv}
\alias{psurv}
\alias{qsurv}
\alias{rcoxph}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
The Survival Distribution
}
\description{
Density, distribution function, quantile function and random variable generation for a survival distribution with a provided hazard function or cumulative hazard function 
}
\usage{
  dsurv(x, h0 = NULL, H0 = function(x){x}, log=FALSE)
  psurv(q, h0 = NULL, H0 = function(x){x}, low.tail=TRUE, log.p=FALSE) 
  qsurv(p, h0 = NULL, H0 = function(x){x}, low.tail=TRUE)
  rsurv(n, h0 = NULL, H0 = function(x){x})
  rcoxph(n, h0 = NULL, H0 = function(x){x}, lp = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x, q}{vector of quantiles.}
  \item{p}{vector of probabilities.}
  \item{n}{number of observations.}
  \item{h0}{hazard function, default is h0 = NULL.}
  \item{H0}{cumulative hazard function, default is H0(x) = x.}
  \item{lp}{linear predictor for rcoxph, H(x) = H0(x)exp(lp).}
  \item{log, log.p}{logical; if TRUE, probabilities p are give as log(p).}
  \item{low.tail}{logical; if TRUE, probabilities are P[X < or = x] otherwise, S(x) = P[X>x].}
}
\details{
If \{ h0 \} or \{ H0 \} are not specified, they assume the default values of h0(x) = 1 and H0(x) = x, respectively. 

The survival distribution function is given by,

  S(x) = exp(-H0(x)), 

where H0(x) is the cumulative hazard function. Only one of h0 or H0 can be specified, if h0 is given, then H0(x) = integrate(h0, 0, x, subdivisions = 500L)

To generate Cox PH survival time, use 
 
 u = exp(-H(t)*exp(lp))

then, -log(u)*exp(-lp) = H(t). Find t such that H(t) = -log(u)exp(-lp).
}
\value{
%%  ~Describe the value returned
\{ dsurv \} gives the density h(x)/S(x), \{ psurv \} gives the distribution function, \{ qsurv \} gives the quantile function,  \{ rsurv \} generates random survival time, and \{ rcoxph \} generates random survival time with Cox proportional hazards model.

The length of the result is determined by n for rsurv and rcoxph.
}
\references{
Johnson, N. L., Kotz, S. and Balakrishnan, N. (1995). Continuous Univariate Distributions, volume 1. Wiley, New York.
}
\author{
Bingshu E. Chen
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{Distributions}} for other standard distributions, including \code{\link{dweibull}} for the Weibull distribution.
}
\examples{
#### use qsurv to generate quantiles for weibull distribution
H1 = function(x) x^3
qsurv(seq(0.1, 0.9, 0.2), H0 = H1) ### shall be the same as
qweibull(seq(0.1, 0.9, 0.2), 3)
#### to get random survival time from the cumulative hazard function H1(t)
rsurv(15, H0 = H1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
\keyword{ Survival distribution }
\keyword{ Cox PH random variable }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
