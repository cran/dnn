#  activation function and its derivative
sigmoid  = function(x) { return(1/(1+exp(-x))) }
elu      = function(x) { return(ifelse(x>0, x, exp(x)-1)) }
relu     = function(x) { return(ifelse(x>0, x, 0)) }
lrelu    = function(x) { return(ifelse(x>0, x, 0.1*x)) }
idu      = function(x) { return(x) }

dsigmoid = function(y) { return(y*(1-y)) }             # here y = sigmoid(w*x)
delu     = function(y) { return(ifelse(y>0, 1, y+1)) } # here y = elu(w*x)
drelu    = function(y) { return(ifelse(y>0, 1, 0)) }
dlrelu   = function(y) { return(ifelse(y>0, 1, 0.1)) }
dtanh    = function(y) { return(1-(y)^2) }             # tanh is an R function for hyperbolic
                                                       # here y = tanh(w*x) 
didu     = function(y) { return(1+y-y) }


### dNNmodel
dNNmodel = function(units, activation=NULL, input_shape=NULL, 
           type=NULL, N=NULL, Rcpp=TRUE, optimizer = c("momentum", "nag", "adam")) {
  actList  = c("sigmoid", "relu", "elu", "lrelu", "tanh", "idu")
  optm = match.arg(optimizer);
  
  n_layers = length(units) 
  if(units[n_layers] != 1) 
    warning("The last layer have size > 1. A multi-output model is used.\n")
  p1 = input_shape + 1
  
  params = list()
  params[[1]]= matrix(runif(p1*units[1], -1, 1), p1, units[1])
  if(n_layers>1) for(i in 2:n_layers) {
    p1 = units[i-1]+1
    params[[i]] = matrix(runif(p1*units[i], -1, 1), p1, units[i])
  }
  if(is.null(activation)) activation = rep('relu', n_layers)
  if(is.null(type)) type = rep('Dense', n_layers)
  drvfun = sapply(activation, function(x) paste('d', x, sep=''))
  act.n = match(activation, actList)
  model = list(units = units, activation = activation, drvfun = drvfun, 
          input_shape=input_shape, params = params, type = type, 
          n_layers=n_layers, N = N, act.n = act.n, Rcpp=Rcpp, optimizer = optm)
  class(model) = 'dNNmodel'
  return(model)
}

print.dNNmodel = function(x, ...) {
  model = x
  units = model$units
  if(is.null(model$N)) N = 'NULL' else N = model$N
  n_layers = model$n_layers
  cat('Model: "sequential"\n')
  cat('________________________________________________________________________________\n')
  cat('Layer (type)                        Output Shape                    Param #     \n')
  cat('================================================================================\n')
  params.total = 0
  for(i in 1:n_layers) {
    if(i==1) params.n = model$input_shape*units[1] 
    else params.n = units[i-1]*units[i]
    params.total = params.total+params.n

    cat(model$type[i], '                              ')
    shape = paste('(', N, ',', units[i],')', sep='')
    shape.len = nchar(shape)
    cat(shape)
    for(j in 1:(32-shape.len)) cat(' ')
    cat(params.n, '\n')
    if(i<n_layers) 
         cat('________________________________________________________________________________\n')
    else cat('================================================================================\n')
  }
  cat("Total params:", params.total, '\n')
  cat("Trainable params:", params.total, '\n')
  cat("Non-trainable params: 0\n")
}

summary.dNNmodel = function(object, ...) {
  print(object)
}

### melt the data.frame matrix to a long format
.melt_matrix = function(d) {
  lmt = reshape(d, idvar = "row", ids = row.names(d), v.names = "value",
          times = names(d), timevar = "column",
          varying = list(names(d)), direction = "long")
  return(lmt)
}

plot.dNNmodel = function(x, ...) {
  mx = x$params
  p  = length(mx)-1
  for(i in 1:p) {
    mi = data.frame(mx[[i]])
    di = .melt_matrix(mi)
    #print(li)
    g = ggplot(di, aes(di$row, di$column)) + geom_tile(aes(fill=di$value), colour="white") 
    g = g + scale_fill_gradient(low = "blue", high = "red")
    print(g)
    readline("Press entre to continue...\n")
  }
}

###############################################
### forward propagation of the deep NN ########
### X      : input matrix, 
### params : a list of the weight matrics
### cache  : return a list of each input layer
###############################################
fwdNN = function(X, model){
  ### Use C++ code (fwdNN2) to speed up the program
  if(model$Rcpp) return(fwdNN2(X, model))

  ### the following code is usefule if Rcpp is not available.
  params   = model$params
  n_layers = model$n_layers
  n = nrow(X)
  activation = model$activation
  if(model$input_shape != ncol(X)) stop("Incorrect input shape size")

  #if(is.null(activation)) activation = rep('sigmoid', n_layers - 1)
  if(length(activation) != (n_layers)) stop("Activation shall be the same size as params")
  x0 = rep(1, n)  # the intercept nx1

  #cache stores all input layers, including X
  cache = list()  

  A = cbind(x0, X)  ### add intercept
  cache[[1]] = A
  if(n_layers>1) for(i in 2:(n_layers)) {
    A = cbind(x0, do.call(activation[i-1], list(x=A%*%params[[i-1]])))
    cache[[i]] = A
  }
  ### do not add intercept to the last layer
  cache[[n_layers+1]] = do.call(activation[n_layers], list(x=A%*%params[[n_layers]]))
  return(cache)
}

##################### bwdNN() #####################################################
### find the derivative with respect to weight parameter W using back propagation
### can be repaced by a loop over the layers
### for most cost function, the derivative has the form of 
### t(A_{k-1})%*%dy_k, where X_{k-1} is the input layer of the 
### previous layer and dy_k is the derivative d(cost)/d(output_layer)
### times the derivative of the activation fuction at current layer

#### back propagation to find dL/dW_i #############################################
### dy    : derivative of L with respect to output layer yp
### cache : output cache from the forward propagation 
### params: a list of weight matrics
### dW    : a list of dL/dW_i
###################################################################################
bwdNN = function(dy, cache, model) {
  ### Use C++ code (bwdNN2) to speed up the program
  if(model$Rcpp) return(bwdNN2(dy, cache, model))

  ### the following code is usefule if Rcpp is not available.
  params     = model$params
  activation = model$activation
  drvfun     = model$drvfun
  n          = length(dy)

  ############ batch_size is better used outside of the bwdNN() function
  #if(is.null(batch_size)) batch_size = n
  #if (batch_size > n) stop("Error: batch size too large")
  #else if(batch_size==n) idx = 1:n
  #else if(batch_size < n) idx = sample(1:n, batch_size)
  #dy = dy[idx]
  # use drop = FALSE to prevent reducing the last layer to a vector
  #cache = mapply(function(x) x[idx, , drop = FALSE], cache) 

  n_layers = model$n_layers
  dW = list()

################ these code is old and will be removed in 2024 ##########
#  A1 = cache[[n_layers-2]]; A1 = A1[idx, ]
#  A2 = cache[[n_layers-1]]; A2 = A2[idx, ]
#  px = cache[[n_layers  ]]; px = px[idx]
#
#  Delta = dy[idx]
#  Delta = Delta*do.call(drvfun[n_layers-1], list(y=px))
#
#  dW[[n_layers-1]] = t(A2)%*%Delta 
#
#  for(i in (n_layers-2):1) {
#    Delta = Delta%*%t(params[[i+1]])
#    Delta = Delta*do.call(drvfun[i], list(y=A2))
#    Delta = Delta[, -1]  # remove the intercept
#    dW[[i]] = t(A1)%*%Delta
#    A2 = A1
#    if(i>1) {A1 = cache[[i-1]]; A1 = A1[idx, ]}
#  }
####################### to be removed in 2024 ###############

#### update code for bwdNN
  ## first Delta
  Delta = dy
  Delta = Delta*do.call(drvfun[n_layers], list(y=cache[[n_layers+1]]))
  dW[[n_layers]] = -t(cache[[n_layers]])%*%Delta

  ## update Delta
  for(i in (n_layers-1):1) {
    Delta = Delta%*%t(params[[i+1]])
    Delta = Delta*do.call(drvfun[i], list(y=cache[[i+1]]))
    Delta = Delta[, -1]   ### remove the intercept
    
    ## Calculate dw_i
    dW[[i]] = -t(cache[[i]])%*%Delta
  }
  return(dW)
}

### the predict method apply the dnn model to x with forward propagation 
### x can not be missing
predict.dNNmodel = function(object, x, ...) {
  n_layers = object$n_layers+1
  cache = fwdNN(x, object)
  return(cache[[n_layers]])
}

dnnControl = function(loss = c("mse", "cox", "bin", "log", "mae"), epochs = 300,
  batch_size = 64, verbose = 0, lr_rate = 0.0001, alpha = 0.5, lambda = 1.0, 
  epsilon = 0.01, max.iter = 100, censor.group = NULL, weights = NULL) {
  
  loss = match.arg(loss)
  lossList = c("mse", "cox", "bin", "log", "mae")
  n_loss = match(loss, lossList)
  if(is.na(n_loss)) stop(paste("loss function", loss, "is not supported."))

  list(epochs = epochs, n_loss = n_loss, loss = loss, batch_size = batch_size, 
      verbose = verbose, lr_rate = lr_rate, alpha = alpha, lambda = lambda, 
      epsilon = epsilon, max.iter = max.iter, cGroup = censor.group, 
      weights = weights)
}

### safe to remove
# deepAFTcontrol = function(epochs = 30, batch_size = 64, verbose = 0, 
#       epsilon = NULL, max.iter = 50, lr_rate = 0.0001, alpha = 0.5, lambda = 1.0, 
#       censor.group = NULL) {
#   
#   list(epochs = epochs, batch_size = batch_size, verbose = verbose, 
#        epsilon = epsilon, max.iter = max.iter, lr_rate = lr_rate, alpha = alpha, 
#        lambda = lambda, cGroup = censor.group)
# }

#### This program is mainly for bwdCheck, otherwise, call 
#### .getCost(y, yh, loss) directly
.dnnCost=function(X, y, new_params, model, loss) {
  model$params = new_params
  cache = fwdNN(X, model)
  yh    = cache[[model$n_layers+1]]
  cost  = .getCost(y, yh, loss = loss)$cost
  return(cost)
}

### to run a bwdNN check, use 
### yhat  = fwd(x, model)
### ry    = rnorm(length(yhat), mean = yhat)
### dy    = ry - yhat
### dW    = bwdNN(dy, x, model)
### dWnum = bwdCheck(x, y, model)
bwdCheck = function(X, y, model, h = 0.0001, loss = 'mse') {
  dW = NULL;
  params = model$params
  for(k in 1:(model$n_layers)) {
    sz = dim(params[[k]])
    dp = matrix(NaN, sz[1], sz[2])
    for(i in 1:sz[1]) {
      for(j in 1:sz[2]) { 
        pm1 = params; pm2 = params
        pm1[[k]][i, j] = params[[k]][i, j] - h
        pm2[[k]][i, j] = params[[k]][i, j] + h
        cost1 = .dnnCost(X, y, pm1, model, loss)
        cost2 = .dnnCost(X, y, pm2, model, loss)
        dp[i, j] = (cost2-cost1)/(2*h)
      }
    }
    dW[[k]] = dp
  }
  return(dW)
}
