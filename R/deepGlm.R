#### deep learning for Surv
#deepGlm = function(x, ...) UseMethod("deepGlm")

deepGlm = function(formula, model, family = c("gaussian", "binomial", "poisson"),
                           data, epochs = 200, lr_rate = 0.0001, 
                           batch_size = 64, alpha = 0.7, lambda = 1.0, 
                           verbose = 0, weights = NULL, ...) {
  family = match.arg(family)
  
  if (missing(data)) 
    data = environment(formula)
  
  mf = model.frame(formula=formula, data=data)
  
  x = model.matrix(attr(mf, "terms"), data = mf)
  varNames = colnames(x)
  ### remove intercept term
  x = x[, -1]
  
  sdx = apply(x, 2, sd)
  if(max(sdx)>10) warning("Variance of X is too large, please try xbar = scale(x)")
  
  y = model.response(mf)
  
  f.name = family
  y.mean = mean(y); y.sd = sd(y)
  #null.dev = switch(f.name, 
  #                  binmial  = -2*sum(dbinom(y, m, p0, log=TRUE)), 
  #                  gaussian = -2*sum(dnorm((y-y.mean)/y.sd, log=TRUE)), 
  #                  #gaussian = -2*sum(dnorm(y, y.mean, y.sd, log=TRUE)), 
  #                  poisson  = -2*sum(dpois(y, lambda = y.mean, log=TRUE)))
  null.dev = glm(y~1, family = family)$null.deviance
  
  loss = switch(family, binomial = 'bin', gaussian = 'mse', poisson = 'log')
  ### deal with sample weight
  n = nrow(x)
  p = ncol(x)
  df0 = n-1
  if(is.null(weights)) weights = matrix(rep(1, n), nrow = n)
  
  control = dnnControl(loss= loss, epochs = epochs, lr_rate = lr_rate, 
                       alpha = alpha, lambda = lambda, batch_size = batch_size, 
                       verbose = verbose, weights = weights)
  y  = as.matrix(y)
  fit  = dnnFit(x, y, model, control)
  class(fit) = c("deepGlm", "glm")
  fit$linear.predictors = fit$lp
  fit$call = match.call()
  fit$logLik = -fit$cost
  #fit = deepGlm.default(x, y, model, control)
  fit$family = do.call(family, list())
  fit$df.null = df0
  fit$df.residuals = df0 - p
  fit$rank = p
  fit$null.deviance = null.dev
  fit$deviance = -2*fit$logLik
  fit$aic = fit$deviance + 2*p
  fit$varNames = varNames
  fit$sigma = y.sd
  fit$x = x
  fit$y = y
  return(fit)
}

print.deepGlm = function(x, ...) {
  object = summary(x)
  print(object)
}

print.summary.deepGlm = function(x,...) {
  family = (x$family)$family
  
  cat("\nSummary of deepGlm: deep mixed glm models\n")
  cat("\nCall:\n")
  print(x$call)
  
  cat("\nDeviance Residuals:\n")
  drsum = quantile(x$deviance.resid) #[c(1:3, 5:6)]
  names(drsum) = c("Min.", "1Q", "Median", "3Q", "Max.")
  print(drsum)
  
  tab1 = x$coefficients
  cat("\nCoefficients for the first layer:\n")
  print(tab1, digits = 4)
  
  dv0 = round(x$null.deviance, digits = 3)
  dvr = round(x$deviance, digits = 3)
  sgm = round(x$sigma, digits = 3)
  
  cat("\n(Dispersion parameter for", family, "family taken to be:", sgm,")\n\n")
  cat("    Null deviance:", dv0, "on", x$df.null,     "degree of freedom\n")
  cat("Residual deviance:", dvr, "\n") #, x$df.residual, "degree of freedom\n")
  cat("AIC: ", x$aic, "\n")
  if(!is.null(x$cindex))cat("Concordance index:", round(x$c.index*10000)/10000, "\n\n")
}

summary.deepGlm = function(object, ...) {
  p = object$rank
  
  keep <- match(c("aic", "call", "terms", "deviance", "family", 
                  "contrasts", "df.null", "df.residual", "null.deviance", 
                  "sigma", "iter", "na.action"), names(object), 0L)
  tab1 = object$model$params[[1]]
  rownames(tab1) = object$varNames
  cindex = concordance(object$y~object$fitted.values)
  c.index = cindex$concordance
  
  ans <- c(object[keep], list(deviance.resid = residuals(object, type="deviance"), 
            cindex = cindex, c.index = c.index, coefficients = tab1))
  
  class(ans) = "summary.deepGlm"
  return(ans)
}

residuals.deepGlm = function(object, type = c("deviance", "partial"), ...) {
  type = match.arg(type)
  r = object$residuals
  d.res = sign(r) * sqrt(object$dvi)
  
  res = switch(type, deviance=d.res, partial=r)
  return(res)
}


