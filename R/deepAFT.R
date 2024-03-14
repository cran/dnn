#### deep learning for AFT
deepAFT = function(x, ...) UseMethod("deepAFT")

deepAFT.formula = function(formula, model, data, control = list(...), 
        method = c("BuckleyJames", "ipcw", "transform"), ...) {
  if (missing(data)) 
    data = environment(formula)

  mf = model.frame(formula=formula, data=data)
  method = match.arg(method)

  x = model.matrix(attr(mf, "terms"), data = mf)
  ### remove intercept term
  x = x[, -1]
  
  sdx = apply(x, 2, sd)
  if(max(sdx)>10) warning("Variance of X is too large, please try xbar = scale(x) ")
  
  y = model.response(mf)

  if (!inherits(y, "Surv")) 
    stop("Response must be a survival object")

  type = attr(y, "type")
  if (type == "counting") 
    stop("start-stop type Surv objects are not supported")
  if (type == "mright" || type == "mcounting") 
    stop("multi-state survival is not supported")

  if (missing(control)) control = dnnControl(...)
    else control =  do.call("dnnControl", control)

  fit = switch(method, BuckleyJames=deepAFT.default(x, y, model, control), 
                       ipcw      = deepAFT.ipcw(x, y, model, control),
                       transform = deepAFT.trans(x, y, model, control))
  return(fit)
}

deepAFT.default = function(x, y, model, control, ...) {
  batch_size = control$batch_size
  epochs = control$epochs
  verbose = control$verbose
  max.iter= control$max.iter
  epsilon = control$epsilon
  lr_rate = control$lr_rate
  alpha   = control$alpha
  lambda  = control$lambda
  
  time = y[, 1]

  status = y[, 2]
  n = length(status)
  max.t = max(time)
  
  ### deal with issue when subject with max.t is censored
  cidx = ifelse(time==max.t & status == 0, 1, 0)
  count = sum(cidx)
  if(count > 2) {
    if(verbose) cat("Note: ", count, "subjects with censoring time = max.t were imputed!\n")
    time[cidx] = runif(count, max.t, max.t + 1)
    status[cidx] = rbinom(count, 1, 0.5)
    y = Surv(time, status)
    max.t = max(time)
  }
  
  if(is.null(epsilon)) epsilon = 0.0005
  id = 1:n
  dat = data.frame(cbind(id = id, time = time, status = status))
  ep = 1
  dati = dat
  ipt = .imputeKM(dat)*ep
  ipt0 = ipt
  mean.ipt = mean(log(ipt))
  
  dnnCtl = dnnControl(epochs = epochs, lr_rate = lr_rate, alpha = alpha, 
           lambda = lambda, weights = rep(1, n), batch_size = batch_size,
           epsilon = epsilon, loss = 'mse')
  convergence = FALSE
  history = NULL
  for(k in 1:max.iter) {
    ###lgy = log(T), with T = imputed time (ipt)
    lgt = log(ipt) - mean.ipt
    #print(summary(lgt))

    dnnCtl$lr_rate = lr_rate/k
    x = as.matrix(x); lgt = as.matrix(lgt)
    fit = dnnFit(x, lgt, model, dnnCtl)
    #print(as.vector(fit$history))

    ep0 = ep
    lp = predict(fit$model, x)+mean.ipt
    ep = exp(lp)
    history = c(history, fit$history)
    logLik = fit$logLik

    ### do imputation for censoring time
    et = dat$time/ep
    ### restrict rescaled time to be less than max.t
    et = ifelse(et < max.t, et, max.t)
    #dati = data.frame(cbind(id = id, time = et, status = status))
    dati = data.frame(cbind(id, et, status))
    colnames(dati) = c("id", "time", "status")
    ipt = .imputeKM(dati)*ep
    ### restrict imputed time to be less than max.t
    ipt = ifelse(ipt < max.t, ipt, max.t)
    resid = (log(ipt) - lp)
    if(verbose == TRUE) cat('MSE = ', mean(resid^2), " ")

    #check convergence
    dif.ep = mean(abs(ep-ep0))/max.t
    #cat(",  epsilon = ", dif.ep, "\n")
    if(dif.ep < epsilon) {
      convergence = TRUE
      break
    }
    model = fit$model
  }
  if(verbose) {
    if(!convergence) cat("Maximum iterations reached before converge!\n") 
    else cat('Algorithm converges after ', k, 'iterations!\n')
  }
  ### create outputs
  object = list(x = x, y = y, model = model, mean.ipt = mean.ipt,
      history = history, logLik = logLik, 
      predictors = lp, risk = exp(-lp), iter = k, method = "Buckley-James")
  class(object) = c('deepAFT', 'dSurv')
  return(object)
}

deepAFT.ipcw = function(x, y, model, control, ...){
  batch_size = control$batch_size
  epochs  = control$epochs
  verbose = control$verbose
  lr_rate = control$lr_rate
  alpha   = control$alpha
  lambda  = control$lambda
  epsilon = control$epsilon
  
  #cGroup  = control$cGroup
  time   = y[, 1]
  status = y[, 2]
  n = length(status)
  
  batch_size = n  ###no mini batch for IPCW
  
  # fit a KM curve for censoring.
  Gfit = survfit(Surv(time, 1-status)~1)

  #sg = Gfit$surv
  #smin = min(sg[sg>0]) #set surv to a small number if the last obs fails.
  #sg = ifelse(sg > 0, sg, smin)
  #G = status/.appxf(sg, x=Gfit$time, xout = time)

  G = status/.ipcw(time, status)

  lgt = log(time)
  mean.ipt = mean(lgt)
  lgt = lgt-mean.ipt
  
  dnnCtl = dnnControl(epochs = epochs, lr_rate = lr_rate, alpha = alpha,
             lambda = lambda, epsilon = epsilon, weights = G, loss = 'mse', 
             batch_size = batch_size)

  x = as.matrix(x); lgt = as.matrix(lgt)
  fit = dnnFit(x, lgt, model, dnnCtl)

  ### update model
  model = fit$model
 
  ### predictors
  lp = predict(model, x) + mean.ipt
  history = fit$history
  
  ### create outputs
  object = list(x = x, y = y, model = model, history = history, 
                logLik = fit$logLik, mean.ipt = mean.ipt, 
                predictors = lp, risk = exp(-lp), method = "ipcw")
  class(object) = c('deepAFT', 'dSurv')
  return(object)
}

# fit km curve for censoring, set surv to small number if last obs fails.
# transformation based on book of Fan J (1996, page 168)
.Gfit = function(time, status, cGroup = NULL) {
  Gfit = survfit(Surv(time, 1-status)~1)
  St = Gfit$surv
  tm = Gfit$time
  St = ifelse(St > 0, St, 1e-5)
  Gt = 1/.appxf(St, x=tm, xout = time)

  # Integrate from 0 to t of 1/G(u)
  dt = diff(c(0, tm))
  iG = cumsum(1/St*dt)
  iGt = .appxf(iG, x=tm, xout = time)

  a = min(((iGt - time)/(time*Gt-iGt))[status==1], na.rm=TRUE)
  if (a > 1) a = 1
  if (a < -1) a = -0.99
  #cat('a = ', a, '\n')

  phi2 = (1+a)*iGt
  phi1 = phi2 - a*time*Gt

  tx = ifelse(status>0, phi1, phi2)
  return(tx)
}

deepAFT.trans = function(x, y, model, control, ...){
  batch_size = control$batch_size
  epochs  = control$epochs
  verbose = control$verbose
  lr_rate = control$lr_rate
  alpha   = control$alpha
  lambda  = control$lambda
  epsilon = control$epsilon
  
  time = y[, 1]
  status = y[, 2]
  
  ### initial values for mu
  mu = mean(time)
  #mu = 1
  n = nrow(x)

  history = NULL
  for(i in 1:3){
    tx = time/mu
    tx2 = .Gfit(tx, status)*mu
    lgt = log(tx2)
    #print(head(lgt))
    mean.ipt = mean(lgt, na.rm = TRUE)
    #print(mean.ipt)
    lgt = lgt-mean.ipt
  
    dnnCtl = dnnControl(epochs = epochs, lr_rate = lr_rate, alpha = alpha,
               lambda = lambda, epsilon = epsilon, weights = rep(1, n), 
               batch_size = batch_size, loss = 'mse')

    x = as.matrix(x); lgt = as.matrix(lgt)
    fit = dnnFit(x, lgt, model, dnnCtl)

    ### update model
    model = fit$model

    #predictors
    lp = predict(model, x) + mean.ipt
    #print(head(lp))

    mu = exp(lp)
    history = c(history, fit$history)
  }
  ### create outputs
  object = list(x = x, y = y, model = model, mean.ipt = mean.ipt, 
                history = history, predictors = lp, risk = exp(-lp), 
                method = "transform")
  class(object) = c('deepAFT', 'dSurv')
  return(object)
}

plot.deepAFT = function(x, type = c('predicted', 'residuals', 'baselineKM'), ...) {
  type = match.arg(type)
  time = x$y[, 1]
  log.time  = log(time)
  if (type == 'predicted') {
    predicted = x$predictors
    plot(log.time, predicted, xlab = 'Log survival time', ylab = 'Predicted log survival time')
    abline(0, 1, lty = 2)
  } else if(type == 'residuals') {
    resid = x$residuals
    plot(log.time, resid, xlab = 'Log survival time', ylab = 'Residuals of linear predictors')
    abline(0, 0, lty = 2)
  } else if(type == 'baselineKM') {
    sfit = survfit(x)
    plot(sfit)
    title('Baseline KM curve for T0 at X = 0')
  }
}

print.deepAFT = function(x, ...) {
  object = summary(x)
  print(object)
}

print.summary.deepAFT = function(x, ...) {
  cat("Deep AFT model with", x$method, 'method\n\n')
  cat("Summary of predicted values of mu, location exp(mu) and martingale residuals:\n")
  out = data.frame(cbind(predictors = x$predictors, locations = x$locations))
  colnames(out) = c('predictors', 'locations')
  if(!is.null(x$residuals)) out$residuals = x$residuals

  print(t(apply(out, 2, summary)), digits = 3)
  if(!is.null(x$cindex))cat("Concordance index:", round(x$c.index*10000)/10000, "\n\n")
  
  cat("for n = ", length(out[, 1]), 'observation(s).\n')

  cat("\nDistribution of T0 = T/exp(mu) for the training data:\n")
  print(x$sfit)
  
  logLik = -x$logLik
  cat('log Lik.', logLik, '\n')
}

summary.deepAFT = function(object, ...) {
  risk = as.vector(object$risk)
  y = object$y
  lp = object$predictors
  locations = 1/risk
  #lp = object$predictors
  sfit = survfit(object)

  cindex = concordance(y~lp)
  c.index = cindex$concordance
  
  resid = residuals(object, type = 'm')
  newy = FALSE
  temp = list(predictors = object$predictors, locations = locations, sfit = sfit, 
              cindex = cindex, c.index = c.index, residuals = resid, 
              method = object$method, logLik = object$logLik, newy = newy)
  class(temp) = "summary.deepAFT"
  return(temp)
}

#### impute KM for AFT
### impute censoring time
## st is a n x 2 matrix, with st[ ,1] as time and st[, 2] as survival function
.imputeFun = function(tc, st) {
  sc = st[st[, 1] > tc, ]
  if(is.vector(sc)) return(sc[1])
  
  ## P(T>tc)
  sm = sum(sc[, 2])
  
  ##conditional probability mass function
  pmf  = sc[, 2]/sm
  ## imputed survival time
  ipt = sum(sc[, 1]*pmf)
  return(ipt)
}

.imputeKM = function(dat) {
  sf = survfit(Surv(time, status)~1, data = dat)
  sv = sf$surv
  #sv[length(sv)] = 0
  st = cbind(sf$time, -diff(c(1, sv)))

  idc = dat[dat$status == 0, 1]
  ipt = dat$time
  
  d.max = max(dat$time[dat$status > 0])
  c.max = max(dat$time[dat$status < 1])
  for (i in idc) {
    tc = dat$time[i]
    if (tc < d.max)
      ipt[i] = .imputeFun(tc, st)
    else ipt[i] = c.max
  }
  return(ipt)
}


### Find IPCW for factor x (with 5 or less levels).
.ipcw = function(time, status, x=NULL) {
  n = length(status)
  event = 1 - status
  if(is.null(x)) x = rep(1, n)

  # Fit a Cox model is input 'x' is a matrix;
  if(!is.vector(x)) {
    cfit = coxph(Surv(time, event)~x)
    gt = exp(-predict(cfit, type = 'expected'))
    smin = min(gt[gt>0]) #set surv to a small number if the last obs fails.
    gt = ifelse(gt>0, gt, smin)
    return(gt)
  }
  # Fit a nonparametric model if input x is a factor or a vector 
  if(length(unique(x))>20) xf = cut(x, 5)
  else xf = as.factor(x)
  xn = as.numeric(xf)
  max.xn = max(xn)
  gt = rep(NaN, n)
  if(max.xn > n/20) stop("Too many censoring groups. Please use 5 or less censoring groups.")
  for(i in 1:max.xn) {
    idx = (xn == i)
    gf = survfit(Surv(time, event)~xn, subset = idx)
    ti = time[idx]
    tg = gf$time
    sg = gf$surv
    smin = min(sg[sg>0]) #set surv to a small number if the last obs fails.
    sg = ifelse(sg > 0, sg, smin) 
    si = .appxf(sg, tg, ti)
    gt[idx] = si
  }
  return(gt)
}
