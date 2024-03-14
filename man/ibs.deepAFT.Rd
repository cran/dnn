\name{ibs}
\alias{ibs}
\alias{ibs.deepAFT}
\alias{ibs.default}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculate integrated Brier Score for deepAFT 
}
\description{
The function ibs is used to calculate integrated Brier Score for deepAFT.
}
\usage{
ibs(object, ...)
### To calculate Brier score for the original fitted data
\method{ibs}{default}(object, ...)
### To calculate Brier score for new data with new outcomes
\method{ibs}{deepAFT}(object, newdata=NULL, newy = NULL, ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{the results of a deepAFT fit.}
  \item{newdata}{optional argument, if no null, new data and new y will be used for calculation.}
  \item{newy}{optional argument, used together with new data.}
  \item{\dots}{other unused arguments.}
}
\details{
  ibs is called to calculate integrate Brier score for the deepAFT model \code{\link{deepAFT}}.
}
\value{
%%  ~Describe the value returned
A list contains the integrate Brier score and the Brier score is returned:
%%  If it is a LIST, use
  \item{ibs}{Integerate Brier score}
  \item{bs}{Brier score}
}

\author{Bingshu E. Chen}
\seealso{
  \code{\link{deepAFT}}
}
%\examples{}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ Integrated Brier Score }
