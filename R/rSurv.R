#.appxf = function(y, x, xout){ approx(x,y,xout=xout,rule=2)$y }

# psurv returns p = 1 - S(t) = 1-exp(-H(t)), where t = q
# it returns p = 1 - S(t) if low.tail = FALSE
psurv = function(q, h0 = NULL, H0 = function(x){x}, low.tail=TRUE, log.p=FALSE) {
  #print(q)
  H = function(t) { integrate(h0, 0, t, subdivisions = 500L)$value }
  if(!is.null(h0)) Ht = vapply(q, H, 1)
  else Ht = vapply(q, H0, 1)

  if(low.tail) {
    if (log.p) return(Ht)
    else return(1-exp(-Ht))
  } else {
    if (log.p) return(-Ht)
    else return(exp(-Ht))
  }
}

dsurv = function(x, h0 = NULL, H0 = function(x){x}, log=FALSE) {
  if(!is.null(h0)) h0x = h0(x)
  else {
    epsilon = 1e-5
    h0x = (H0(x+epsilon)-H0(x-epsilon))/(2*epsilon)
  }
  if(sum(h0x<=0)>1) stop("h0(t) must be positive")
  logf = log(h0x) + psurv(x, h0=h0, H0=H0, low.tail=FALSE, log.p=TRUE)
  if(log) return(logf) else return(exp(logf))
}

qsurv = function(p, h0 = NULL, H0 = function(x){x}, low.tail=TRUE) {
  s01 = uniroot(function(x) psurv(x, h0, H0) - 0.01, c(0, 10), extendInt="upX")$root
  s99 = uniroot(function(x) psurv(x, h0, H0) - 0.99, c(0, 10), extendInt="upX")$root
  tx = seq(s01, s99, (s99-s01)/200)
  if(!low.tail) p = 1-p
  Ht = psurv(tx, h0, H0, low.tail = FALSE, log.p = TRUE)
  
  #### .appxf(y=t, x=Ht, x0 = log(1-p))
  qx = .appxf(tx, Ht, log(1-p))
  return(qx)
}

rsurv = function(n, h0 = NULL, H0 = function(x){x}) {
  x = qsurv(runif(n, 0, 1), h0=h0, H0=H0)
  return(x)
}

rcoxph= function(n, h0 = NULL, H0 = function(x){x}, lp = 0) {
  n1 = length(lp)
  if(n1 == 1) lp = rep(lp, n)
  else if(n1 != n) stop("length of lp shall either be 1 or n.")
  
  ###      U = S(T) = exp(-H0(T)exp(lp)) 
  ### ==>  U2 = -log(U)*exp(-lp) = H0(T)
  ### ==>  T = inv.H0(U2)
  u = runif(n, 0, 1)
  u2 = -log(u)*exp(-lp)

  H.body = quote({ psurv(x, h0, H0, log.p = TRUE) })

  u9 = quantile(u2, 0.99)
  #print(u9)
  t9 = try(uniroot(function(x) eval(H.body) - u9, c(0, 10), extendInt="upX")$root)
  #print(t9)
  if (is(t9, "try-error")) stop("Survival time too large, please check H(t) and lp.")

  t0 = seq(0, t9, t9/100)
  Ht = psurv(t0, h0, H0, log.p = TRUE)
  x  = .appxf(t0, Ht, u2)
  return(x)
}

### example
#h = function(t) ht0 = exp(-5 + 2.5*cos(pi*t)+0.5*t^0.3)
#print(qsurv(0.5))
# H1 = function(x) x^3
# qsurv(seq(0.1, 0.9, 0.2), H0 = H1) ### shall be the same as
# qweibull(seq(0.1, 0.9, 0.2), 3)
