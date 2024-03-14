\name{plot}
\alias{plot.deepAFT}
\alias{plot.dNNmodel}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Plot methods in dnn package
}
\description{
Plot function for plotting of R objects in the dnn package.

Several different type of plots can be produced for the deep learning mdels. 
Plot method is used to provide a summary of outputs from "deepAFT", "deepGLM", "deepSurv" and "dnn".

Use "methods(plot)" and the documentation for these for other plot methods.
}
\usage{
\method{plot}{dNNmodel}(x, ...)
\method{plot}{deepAFT}(x, type = c("predicted", "residuals", "baselineKM"), ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{a class of "dNNmodel".}
  \item{type}{type of plot in deepAFT object, "predicted" to plot the linear predicted values, "residuals" to plot residuals, "baselineKM" to plot baseline Kaplan-Meier survival curve.}
  \item{\dots}{other options used in plot().}
}
\details{
  plot.deepAFT is called to plot the fitted deep learning AFT model.

  plot.dNNmodel is called to plot fitted dnn model

  The default method, plot.default has its own help page. Use methods("plot") to get all the methods for the plot generic.
}
\value{No return value, called to plot a figure.}
\author{Bingshu E. Chen}
\seealso{
 The default method for plot \code{\link{plot.default}}.
\code{\link{glm}}
}
%\examples{}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ deepAFT }
\keyword{ deepGLM }
\keyword{ deepSurv }
\keyword{ plot }
\keyword{ summary }
