#### deep learning for Surv
#deepSurv = function(x, ...) UseMethod("deepSurv")

deepSurv = function(formula, model, data, epochs = 200, lr_rate = 0.0001, 
        batch_size = 64, alpha = 0.7, lambda = 1.0, verbose = 0, 
        weights = NULL, ...) {
  if (missing(data)) data = environment(formula)

  mf = model.frame(formula=formula, data=data)

  x = model.matrix(attr(mf, "terms"), data = mf)
  ### remove intercept term
  x = x[, -1]
  
  sdx = apply(x, 2, sd)
  if(max(sdx)>10) warning("Variance of X is too large, please try xbar = scale(x)")
  
  y = model.response(mf)
  if (!inherits(y, "Surv")) 
    stop("Response must be a survival object")

  type = attr(y, "type")
  if (type == "counting") 
    stop("start-stop type Surv objects are not supported")
  if (type == "mright" || type == "mcounting") 
    stop("multi-state survival is not supported")

  ### deal with sample weight
  n = nrow(y)
  if(is.null(weights)) weights = matrix(rep(1, n), nrow = n)
  else warning("Sample weight is not supported now")

  control = dnnControl(loss= 'cox', epochs = epochs, lr_rate = lr_rate, 
            alpha = alpha, lambda = lambda, batch_size = batch_size, 
            verbose = verbose, weights = weights)
  fit = deepSurv.default(x, y, model, control)
  return(fit)
}

deepSurv.default = function(x, y, model, control, ...) {
  epochs = control$epochs
  idx = order(y[, 1], decreasing = TRUE)
  y1 = y[idx, ]
  x1 = x[idx, ]
  w0 = model$params
  fit  = dnnFit(x1, y1, model, control)

  ### update model
  model = fit$model
  lp = fit$lp 
  logLik = fit$logLik

  ### create outputs
  object = list(x = x1, y = y1, model = model, history = fit$history, 
                predictors = lp, risk = exp(lp), logLik = logLik)
  class(object) = c("deepSurv", "dSurv")
  return(object)
}

print.deepSurv = function(x, ...) {
  object = summary(x)
  print(object)
}

print.summary.deepSurv = function(x, ...) {
  cat("Summary of predicted values, risk score  and martingale residuals:\n")
  out = data.frame(cbind(predictors = x$predictors, risk = x$risk))
  colnames(out) = c('predictors', 'risk')
  if(!is.null(x$residuals)) out$residuals = x$residuals

  print(t(apply(out, 2, summary)), digits = 3)
  if(!is.null(x$cindex))cat("Concordance index:", round(x$c.index*10000)/10000, "\n\n")
  
  cat("for n = ", length(out[, 1]), 'observation(s).\n')

  cat("\nDistribution of baseline survival time for the training data:\n")
  print(x$sfit)
  
  logLik = -x$logLik
  cat('log Lik.', logLik, '\n')
}

summary.deepSurv = function(object, ...) {
  risk = as.vector(object$risk)
  y = object$y
  lp = object$predictors
  loc= -risk
  sfit = survfit(object)

  cindex = concordance(y~loc)
  c.index = cindex$concordance
  
  resid = residuals.dSurv(object, type = 'm')
  temp = list(predictors = object$predictors, risk = risk, sfit = sfit, cindex = cindex, 
              c.index = c.index, residuals = resid, logLik = object$logLik)
  class(temp) = "summary.deepSurv"
  return(temp)
}
