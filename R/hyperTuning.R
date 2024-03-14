hyperTuning = function(x, y, model, ER = c("cindex", "mse"), 
              method = c('BuckleyJames', 'ipcw', 'transform', 'deepSurv'), 
              lower = NULL, upper = NULL, node = FALSE,
              K = 5, R = 25) {
  p = ncol(x)
  n = nrow(x)
  method = match.arg(method)
  ER     = match.arg(ER)

  if(missing(model)) model = dNNmodel(units = c(8, 6, 1), activation = c("relu", "relu", "idu"), 
                 input_shape = p)
  units = model$units
  n_layers = length(units) 

  if(!is.matrix(y)) y = as.matrix(y, nco = 1)
  
  ### prediction error
  e0 = switch(ER, cindex = 0, mse = 10000)
  p0 = c(e0, 0)
  c0 = NULL
  
  if(is.null(lower)) lower = dnnControl(epochs = 5000, batch_size = 128, epsilon = 1E-3)
  
  epochs = lower$epochs
  batch_size = lower$batch_size
  epsilon = lower$epsilon
  
  alpha.low  = lower$alpha
  lambda.low = lower$lambda
  lr.low     = lower$lr_rate
  
  if(is.null(upper)) {
    alpha.up  = 0.97
    lambda.up = 10
    lr.up     = 0.01
  } else {
    alpha.up  = upper$alpha
    lambda.up = upper$lambda
    lr.up     = upper$lr_rate
  }
  
  i = 0
  while(i < R) {
    alpha   = runif(1, alpha.low,  alpha.up)
    lambda  = runif(1, lambda.low, lambda.up)
    lr_rate = runif(1, lr.low,     lr.up)

    control = dnnControl(lr_rate = lr_rate, alpha = alpha, lambda = lambda, 
                 epochs = epochs, batch_size = batch_size, epsilon = epsilon)

    if(node){
      for(j in 1:(n_layers-1)) model$units[j] = sample(c(2:units[j]), 1)
    }

    #print(control)
    pe = try(CVpredErr(x, y, model, control, method, ER, K))
    if(is(pe, "try-error")) next
    message("Cross validation ", ER, ": ", pe, '\n')
    i = i + 1
    
    better = switch(ER, cindex = (pe[1] > p0[1]), mse = (pe[1] < p0[1]))
    if(better) {
      p0 = pe
      c0 = control
    }
  }
  message("Optimal tuning parameter with CV ", ER, " = ", p0[1], ", SE = ", p0[2], "\n")
  #print(c0)
  if(method == "deepSurv") {
    c0$loss = 'cox'
    c0$n_loss = 2
  }
  if(method == "BuckleyJames") {
    c0$max.iter = 100
    #c0$epochs = 300
  }
  tmp = list(control = c0, model = model)
  return(tmp)
}

CVpredErr = function(x, y, model, control, method = "deepSurv", ER = c('cindex', 'mse'), K = 5) {
  ER = match.arg(ER)
  n = length(x[, 1])
  index = c(0, round(seq_len(K)*n/K))

  if(method == "deepSurv") {
    control$loss = 'cox'
    control$n_loss = 2
    ### mse does not work for deepSurv 
    ER = 'cindex'
  }
  
  J = length(index)
  tmp = rep(0, J-1)
  for (i in 1:(J-1)) {
    sel = (index[i]+1):(index[i+1])
    x0  = x[-sel, ]
    xt  = x[sel, ]
    y0  = y[-sel,  ]
    yt  = y[sel, ]
  
    fit = switch(method,
      BuckleyJames = deepAFT.default(x0, y0, model, control),
      ipcw         = deepAFT.ipcw(x0,  y0, model, control),
      transform    = deepAFT.trans(x0, y0, model, control),
      deepSurv     = deepSurv.default(x0, y0, model, control)
    )
    #print(class(fit))
    
    tmp[i] = switch(ER, 
      cindex       = predict(fit, xt, yt)$c.index,
      mse          = mseIPCW(fit, xt, yt)
    )
  }
  #print(tmp)
  predErr = mean(tmp)
  peSE = sd(tmp)
  return(c(predErr, peSE))
}
