### Approximate function
.appxf = function(y, x, xout){ approx(x,y,xout=xout,rule=2)$y }

### baseline cumulative hazard function and martingale residuals for dSurv
survfit.dSurv = function(formula, se.fit=TRUE, conf.int = .95, ...) {
  #baseline survival function S_T0(t), with all covariates value = 0
  #where T0 = T/exp(mu), or log(T0) = log(T) - mu, where risk = exp(-mu)
  y0 = formula$y
  y0[, 1] = y0[, 1]*formula$risk
  sfit = survfit(y0 ~ 1, se.fit=se.fit, conf.int=conf.int)
  return(sfit)
}

### Residuals of dSurv
residuals.dSurv = function(object, type=c("martingale", "deviance", "coxSnell"), ...) {
  type = match.arg(type)
  sfit = survfit(object, se.fit = FALSE)
  
  time   = object$y[, 1]*object$risk
  status = object$y[, 2]
  
  m = length(sfit$surv)
  ### in case the last subject fails,  S0(t) = 0
  sfit$surv[m] = sfit$surv[m-1]

  # fit survival function at time Ti
  St = .appxf(sfit$surv, x=sfit$time, xout = time)

  # Cox-Snell residual H(T)
  Ht = -log(St)
  
  rr = status - Ht
  drr = sign(rr)*sqrt(-2*(rr+ifelse(status==0, 0, status*log(status-rr))))
  resid = switch(type, martingale = rr, deviance = drr, coxSnell=Ht)
  return(resid)
}

predict.dSurv = function(object, newdata, newy = NULL, ...) {
  result = summary(object)
  sfit = result$sfit
  if(missing(newdata)) {
    return(result)
  }
  else {
    ### if there is new data
    cls = class(object)[1]
    m = object$model
    x = newdata
    ### if x is a numeric vector, change it to matrix
    if(is.null(dim(x))) x = t(as.matrix(x))
    if(cls == 'deepAFT') {
      lp  = predict(m, x) + object$mean.ipt
      risk = exp(-lp)
      icls = 'summary.deepAFT'
    } else if(cls == 'deepSurv') {
      lp = predict(m, x)
      risk = exp(lp)
      icls = 'summary.deepSurv'
    }
    result$predictors = lp
    result$locations  = 1/risk
    result$risk = risk
    result$cindex = NULL
    result$c.index = NULL
    result$residuals = NULL
  }

  if(!is.null(newy)) {
    if(missing(newdata)) stop("Error: newdata cannot missing when there is new y.")
    if(length(newy[, 1]) != length(x[, 1]))
      stop("Error: new y shall have the same subjects as the new data.")

    time   = newy[, 1]  #time
    status = newy[, 2]  #status

    #baseline survival function
    aft.time = risk*time
    sf = .appxf(sfit$surv, x=sfit$time, xout=aft.time)
    sf = ifelse(sf>0, sf, min(sf[sf>0]))
    cumhaz = -log(sf)
    result$residuals = (status - cumhaz)
    
    #cindex = concordance(newy~lp)
    loc = 1/risk
    cindex = concordance(newy~loc)
    
    result$cindex = cindex
    result$c.index= cindex$concordance
    
    class(result) = icls
  }
  return(result)
}

#####Calculate MSE using ICPW for newdata and newy
mseIPCW = function(object, newdata, newy) {
  time = newy[, 1]
  status = newy[, 2]
  m = length(time)
  cls = class(object)[1]
  if (cls == 'survreg') {
    xa = cbind(rep(1, m), newdata)
    yp = exp(xa%*%object$coef)   #*object$scale)
  }
  if(cls == 'coxph') {
    yp = exp(-newdata%*%object$coef)
  }
  if(cls == 'deepAFT') yp = predict(object, newdata)$locations


  # fit km curve for censoring
  G_fit = survfit(Surv(time, 1-status)~1)
  sg = G_fit$surv
  smin = min(sg[sg>0]) #set surv to a small number if the last obs fails.
  sg = ifelse(sg>0, sg, smin)
  G = .appxf(sg, x=G_fit$time, xout = time)
 
  #print(head(cbind(yp, time, status)))
  #cat('mean = ', mean(yp), '\n')
  mse = sqrt(mean((log(yp)-log(time))^2*status/G))
  return(mse)
}
