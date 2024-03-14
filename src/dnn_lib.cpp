#include "dnn.h"

// forward propagation

//[[Rcpp::export]]
List fwdNN2(const arma::mat& X, const List& model) {
  /*function body*/
  int n = X.n_rows;
  arma::mat x0;
  
  x0.ones(n, 1);

  List params = model("params");
  NumericVector activation = model("act.n");

  //int n_layers = params.size(); /*this is not good since it can change */
  int n_layers = model("n_layers");
  List cache(n_layers+1);

  arma::mat c0 = join_rows(x0, X);
  cache[0] = c0;
  for(int i=0; i < n_layers-1; i++) {
    c0 = c0*as<arma::mat>(params[i]);
    c0 = actfun(c0, activation[i]);
    c0 = join_rows(x0, c0);
    cache[i+1] = c0;

    //std::cout<<"i = "<<i<<" active = "<<activation[i]<<"\n";
    //c0.print(std::cout);
  }

  c0 = c0*as<arma::mat>(params[n_layers-1]);
  cache[n_layers] = actfun(c0, activation[n_layers-1]);

  return cache;
}

// backword propagation

//[[Rcpp::export]]
List bwdNN2(const arma::mat& dy, List cache, const List& model) {
  /*function body*/

  arma::mat     Delta  = dy;
  List          params = model("params");
  NumericVector activation = model("act.n");
  CharacterVector actfunct = model("activation");
  int           n_layers   = model("n_layers");
  List          dW(n_layers);

  // element wise multiple to the d actfun();
  Delta = Delta % dactfun(cache[n_layers], activation[n_layers-1]);
  dW[n_layers-1] = -(as<arma::mat>(cache[n_layers-1]).t()*Delta);

  //std::cout<<"i = "<<n_layers<<" actfunct = "<<actfunct[n_layers-1]<<"\n";
  for(int i = n_layers-1; i>=1; i--) {
    //std::cout<<"i = "<<i<<" actfunct = "<<actfunct[i-1]<<"\n";
    Delta = Delta*as<arma::mat>(params[i]).t();

    // element wise multiply;
    Delta = Delta % dactfun(cache[i], activation[i-1]);

    // remove the intercept column, in C++, intercept is in column 0;
    Delta.shed_col(0);
    dW[i-1] = -(as<arma::mat>(cache[i-1]).t()*Delta);
  }
  return(dW);
}


//// SGD optimizer methods, includes momentum, NAG, and AdamG.

//[[Rcpp::export]]
List optimizerMomentum(List V, List dW, List W, double alpha=0.63, double lr=0.0001, double lambda=1.0) {
  /*function: V = a*V-lr*(dW + lambda*W); W = W + V; # for momentum method */
  int w_len = W.size();
  int v_len = V.size();
  arma::mat v, dw, w;
  List nV(w_len), nW(w_len);

  for(int i=0; i<w_len; i++) {
    w  = as<arma::mat>(W[i]);
    dw = as<arma::mat>(dW[i]);

    if(v_len < w_len) v = -lr*(dw + lambda*w);
    else v = alpha*as<arma::mat>(V[i]) - lr*(dw + lambda*w); 
    
    nV[i] = v;
    nW[i] = w + v;
  }
  return Rcpp::List::create(Rcpp::Named("V")=nV,
                            Rcpp::Named("W")=nW);
}

//[[Rcpp::export]]
List optimizerNAG(List V, List dW, List W, double alpha=0.63, double lr=0.0001, double lambda=1.0) {
  /*function: gd = -lr*(dW + lambda*W); V = a*(V + gd); W = W + V + gd; # for the NAG method */
  int w_len = W.size();
  int v_len = V.size();
  arma::mat v, dw, w, gd;
  List nV(w_len), nW(w_len);

  for(int i=0; i<w_len; i++) {
    w  = as<arma::mat>(W[i]);
    dw = as<arma::mat>(dW[i]);
    gd = -lr*(dw + lambda*w);

    if(v_len < w_len) v = gd;
    else v = alpha*(as<arma::mat>(V[i]) + gd);
    
    nV[i] = v; 
    nW[i] = w + v + gd;
  }
  return Rcpp::List::create(Rcpp::Named("V")=nV,
                            Rcpp::Named("W")=nW);
}

/*  Adam optimizer  
### to use Adam (R code): 
##  v1 =.updateAdamV1(dw, v1)
##  v2 =.updateAdamV2(dw, v2)
##  g  = updateAdamG(v1, v2, epoch, learning_rate)
##  w  = w - g
##
##.updateAdamV1 = function(dw, v1, beta1=0.9    ) return(beta1*v1 + (1-beta1)*dw)
##.updateAdamV2 = function(dw, v2, beta2 = 0.999) return(beta2*v2 + (1-beta2)*(dw*dw))
##
##updateAdamG = function(v1, v2, epoch, lr_rate, beta1=0.9, beta2=0.999) {
##  v1 = v1/(1-beta1^epoch);  v2 = v2/(1-beta2^epoch);
##  g  = v1/(sqrt(v2)+1E-6)*lr_rate
##}
*/

//[[Rcpp::export]]
List optimizerAdamG(List V1, List V2, List dW, List W, int epoch, double beta1=0.9, 
                 double beta2=0.999, double lr=0.0001, double lambda=1.0) {
  int w_len = W.size();
  int v_len = V1.size();
  arma::mat v1, v2, dw, w, gd;
  double epsilon = 1E-6;
  List nV1(w_len), nV2(w_len), nW(w_len);

  for(int i=0; i<w_len; i++) {
    w  = as<arma::mat>(W[i]);
    dw = as<arma::mat>(dW[i]);
    gd = -lr*(dw + lambda*w);

    if(v_len < w_len){
      v1 = (1-beta1)*gd;
      v2 = (1-beta2)*(gd % gd);
    } else {
      v1 = as<arma::mat>(V1[i]); v1 = beta1*v1 + (1-beta1)*gd; 
      v2 = as<arma::mat>(V2[i]); v2 = beta2*v2 + (1-beta2)*(gd % gd); 
    }
    nV1[i] = v1; nV2[i] = v2; 
    v1 = v1/(1-pow(beta1, epoch)); v2 = v2/(1-pow(beta2, epoch));
    nW[i]  = w + v1/(sqrt(v2)+epsilon);
  }
  return Rcpp::List::create(Rcpp::Named("V1")=nV1, Rcpp::Named("V2")=nV2, 
                            Rcpp::Named("W")=nW);
}


//[[Rcpp::export]]
List dnnFit2(const arma::mat& X, const arma::mat& y, const List& model, const List& control) {
  int epochs = control("epochs"), n_layers, n_loss, batch_size = 64;
  int n      = X.n_rows;
  double lr  = control("lr_rate"), alpha, cost, lambda;
  alpha      = control("alpha");
  lambda     = control("lambda");
  n_loss     = control("n_loss");
  batch_size = control("batch_size");
  n_layers   = model("n_layers");

  List V(n_layers), W, dW, VW, dc, W0, V0, m0 = model;
  W        = model("params");
  for(int i = 0; i < n_layers; i++) V[i] = zeros(size(as<arma::mat>(W[i])));
  arma::mat dy, dy2, history(epochs, 1);
  arma::mat sample_weight = control("sample_weight");

  //std::cout<<"n_layers = "<<n_layers<<"\n";
  //std::cout<<"size(V0) = "<<V.size()<<"\n";

  int bz = n - batch_size;
  //std::cout<<"bz = "<<bz<<"n = "<<n<<"\n";
  for(int i=0; i < epochs; i++) {
    arma::uvec idx = arma::randperm(n, bz);
 
    //stochastic gradient descent;   
    arma::mat x0 = X, y0 = y, s0 = sample_weight;
    x0.shed_rows(idx);
    y0.shed_rows(idx);
    s0.shed_rows(idx);
    //std::cout<<"x0 = "<<x0<<"\n";

    List cache   = fwdNN2(x0, m0);
    arma::mat yh = cache[n_layers];

    dc = getCost(y0, yh, n_loss, s0);
    arma::mat dy = dc("dy"); 
    cost = dc("cost");
    //std::cout<<"i = "<<i<<"n = "<<n<<"bz="<<bz<<cost<<"\n";

    dW = bwdNN2(dy, cache, m0);
    W  = m0("params");
    VW = optimizerMomentum(V, dW, W, alpha, lr, lambda);

    V = VW("V"); W = VW("W");
    m0("params") = W;
    history(i, 0) = cost;
  }
  
  m0("params") = W;
  List cache   = fwdNN2(X, m0);
  arma::mat yh = cache[n_layers];
  dc = getCost(y, yh, n_loss, sample_weight);
  cost = dc("cost");

  return Rcpp::List::create(Rcpp::Named("model")=m0, Rcpp::Named("logLik")=cost, 
                            Rcpp::Named("history")=history, 
                            Rcpp::Named("lp")=yh);
}

//// helper functions

// cost of different loss functions;
// lossList = c('mse', 'cox', 'bin', 'log');
List getCost(const arma::mat& y, const arma::mat& yh, const int& i, const arma::mat& sample_weight) {
  if(i==0) Rcpp::Rcout<<"loss function is not defined yet\n";
  arma::mat dy, dy2;
  double cost = 0.0;

  if(i==1) {
    dy = (y - yh)%sample_weight; dy2 = dy%dy;
    cost = accu(dy2);
  } else if(i==2) {   //coxph loss, time shall be decreasing sorted 
    arma::mat eb = exp(yh), delta = y.col(1); 
    arma::mat S0 = cumsum(eb), ht = delta/S0, rht = flipud(ht); 
    arma::mat Ht = cumsum(rht); // Nelson Aalen estimate of the cumulative hazard function
    Ht = flipud(Ht);
    cost = -accu(delta%(yh - log(S0)));
    dy   = delta%(ones(size(yh)) - eb%Ht);
  } else if(i>2) Rcpp::Rcout<<"loss function is not defined yet\n";

  return Rcpp::List::create(Rcpp::Named("cost")=cost, 
                            Rcpp::Named("dy")=dy);
}

//                                   1          2       3      4        5       6
//const CharacterVector actList  = c("sigmoid", "relu", "elu", "lrelu", "tanh", "idu")
arma::mat actfun(const arma::mat& x, const int& i) {
  arma::mat y = x;
       if(i==1) y = sigmoid(x);
  else if(i==2) y = relu(x);
  else if(i==3) y = elu(x);
  else if(i==4) y = lrelu(x);
  else if(i==5) y = tanh(x);
  else if(i==6) y = x;
  if(i > 6) Rcpp::Rcout<<"Activation function i = "<<i<<" is not in the defined list\n ";
  return(y);
}

arma::mat dactfun(const arma::mat& y, const int& i) {
  arma::mat x = y;

  if(i==1) x = dsigmoid(y);
  else if(i==2) x = drelu(y);
  else if(i==3) x = delu(y);
  else if(i==4) x = dlrelu(y);
  else if(i==5) x = dtanh(y);
  else if(i==6) x = ones(size(y));
  if(i > 6) Rcpp::Rcout<<"derivative of the activation function is not in the defined list\n ";
  return(x);
}

// additional activation functions

arma::mat elu(const arma::mat& x){
  arma::mat y;
  arma::umat x1 = (x>zeros(size(x)));
  //y = ifelse(x>0, x, exp(x)-1); 
  y   = x1%x + (ones(size(x))-x1)%(exp(x) - 1);
  return(y);
}

arma::mat delu(const arma::mat& y) {
  arma::mat x;
  arma::umat y1 = (y>zeros(size(y)));
  //x = ifelse(x>0, x, exp(x)) where exp(x) = y + 1;
  x   = y1 % ones(size(y)) + (ones(size(y))-y1)%(y+1); 
  return(x);
}

arma::mat lrelu(const arma::mat& x){
  arma::mat y;
  arma::umat x1 = (x>zeros(size(x)));
  //y = ifelse(x>zeros(size(x)), x, x1); 
  y   = x1%x + (ones(size(x))-x1)%x*0.1;
  return(y);
}

arma::mat dlrelu(const arma::mat& y) {
  arma::mat x;
  arma::umat y1 = (y>zeros(size(y)));
  //x = ifelse(x>0, x, exp(x)) where exp(x) = y + 1;
  x   = y1 % ones(size(y)) + (ones(size(y))-y1)*0.1;
  return(x);
}

