\name{msePICW}
\alias{mseIPCW}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Mean Square Error (mse) for a survival Object
}
\description{
Compute Mean Square Error (mse) values for a survival object
}
\usage{
## S3 method for class 'deepAFT'
mseIPCW(object, newdata, newy)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{the results of a model fit using a deepAFT or a survreg function.}
  \item{newdata}{optional new data at which to do predictions. If absent, predictions are for the dataframe used in the original fit.}
  \item{newy}{optional new outcome variable y.}
}
\details{
  predict is called to predict object from a deepAFT \code{\link{deepAFT}} or a survreg model.
  
  IPCW method is used to calcuate the mean square error for censored survival time.
}

\value{
%%  ~Describe the value returned
mseIPCW returns the mse for the predicted survival data.
}
\author{Bingshu E. Chen}
\seealso{
 The default method for predict \code{\link{predict}},
\code{\link{deepAFT}}, \code{\link{survfit.dSurv}}
}
%\examples{}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ MSE }
\keyword{ IPCW }