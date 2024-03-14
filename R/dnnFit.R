.rcumsum=function(x) rev(cumsum(rev(x))) # sum from last to first

.getCost = function(y, yh, loss='mse', weights=NULL) {
  n = length(yh)
  #print(weights)

  if(loss == 'mse') { 
    if(!is.null(weights)) {
      if(length(weights) != n) stop("Error: Wrong size for weights")
    } else weights = 1
    dy = weights*(y-yh); cost = sum(weights*(y-yh)^2)/2
  } else if(loss == 'cox') {
    ### weights is currently not supported in cox model
    delta = y[, 2]  #*weights
    eb = exp(yh)    #*weights
    S0 = cumsum(eb)
    ht = delta/S0
    Ht = .rcumsum(ht)
    
    cost = -sum(delta*(yh-log(S0)))
    dy   =  (delta - eb*Ht)   
    ### this is actually the martingale residual dN_i - H_0(t_i)exp(x_i'beta)
  }
  else if(loss == 'log') {
    y  = y*weights
    yh = yh*weights
    cost = sum(yh - y*log(yh))
    #dy  = y - y/yh   ### (yh-y)/yh
    dy   = (yh-y)/yh
  }
  else if(loss == 'bin') {
    cost = sum((y*log(yh)+(1-y)*log(1-yh)*weights))
    dy   = (y/yh - (1-y)/(1-yh)*weights)
  }
  else if(loss == 'mae') { 
    dy = weights*abs(y-yh); cost = sum(dy) 
    dy = ifelse(dy>0, -1, 1)
  } else { 
    stop(paste("loss function", loss, "is not defiend yet")) 
  }
  return(list(cost = cost, dy = dy))
}

### fit model of y~x with MSE as the loss function
dnnFit = function(x, y, model, control) {
  #batch_size = 10
  #model$Rcpp = FALSE
  
  convergence = FALSE
  batch_size = control$batch_size
  loss    = control$loss
  epochs  = control$epochs
  LR      = control$lr_rate
  weights = control$weights
  alpha   = control$alpha
  lambda  = control$lambda
  verbose = control$verbose

  useRcpp = model$Rcpp
  n = nrow(x)
  n_layers = model$n_layers
  cost_history <- c()
  v  = 0; v1 = 0; v2 = 0
  x0 = x; y0 = y; w0 = weights ### save the original value
  for (i in 1:epochs) {
    lr_rate = LR/(1+i*0.001)
    
    if(batch_size < n) {  ## mini batch
      idx = sample(1:n, n-batch_size)
      x = x0[-idx, ]
      if(is.matrix(y)) y = y0[-idx, ] else y = y0[-idx]
      weights = w0[-idx]
    }
    
    cache = fwdNN(x, model)
    yh =  cache[[n_layers+1]]
    cy = .getCost(y, yh, loss, weights)
    dy = cy$dy
    cost_history[i] = cy$cost

    dW = bwdNN(dy, cache, model)
    W  = model$params

    if(useRcpp) {
      optimizer = model$optimizer 
      if(optimizer == "momentum") { 
        VW = optimizerMomentum(v, dW, W, alpha=alpha, lr=lr_rate, lambda=lambda)
        v  = VW$V
      }
      if(optimizer == "nag") {
        VW = optimizerNAG(v, dW, W, alpha=alpha, lr=lr_rate, lambda=lambda)
        v  = VW$V
      }
      if(optimizer == "adam") {
        VW = optimizerAdamG(v1, v2, dW, W, epoch = i, lr=lr_rate, lambda=lambda)
        v1 = VW$V1
        v2 = VW$V2
      }
      model$params = VW$W
    } else {
      #stop("Plase install Rcpp")
      getV = function(v, dw, w, alpha, lr, lambda) return(alpha*v-lr*(dw+lambda*w))
      v = mapply(getV, v=v, dw=dW, w=W, alpha=alpha, lr=lr_rate, lambda=lambda)
      model$params = mapply('+', W, v)
    }
    #if ((i %% 10 == 0) & (i>100)) cat("Iteration", i, " | Cost: ", cy$cost, "\n")
    if(i>200) {
      dcost = (cost_history[i-1] - cost_history[i])
      if(is.na(dcost)) warning("NA is generated for cost, check hyper parameters.\n")
      #cat(dcost, lr_rate, '\n')
      if(abs(dcost) < control$epsilon) {
        convergence = TRUE
        break
      }
    }
  }
  yh = predict(model, x0)
  #cost = .getCost(y0, yh, loss, weights=rep(1, n))$cost
  cost = .getCost(y0, yh, loss, weights=w0)$cost
  #cost_history[i] = cost
  if(verbose == TRUE) {
    cat(" Final cost: ", cost, "\n")
    if(!convergence) cat("Maximum eopchs reached, dnnFit may not converge, please check!\n")
  }
  fit = list(model = model, history = cost_history, cost = cost, dW = dW, 
             residuals = dy, fitted.values = yh, dvi = dy*dy, 
             lp = yh, logLik = cost, convergence = convergence)
  return (fit)
}
