#include <iostream>
#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

////  activation functions;
arma::mat sigmoid(const arma::mat& x) {arma::mat y = 1.0/(1+exp(-x)); return(y);}
arma::mat relu(   const arma::mat& x) {arma::mat y = max(zeros(size(x)), x); return(y);}
arma::mat elu(    const arma::mat& x); //define below
arma::mat lrelu(  const arma::mat& x); 


//// derivative of the activation function;
arma::mat dsigmoid(const arma::mat& y) {arma::mat x = y%(ones(size(y))-y); return(x);}
arma::mat drelu(   const arma::mat& y) {arma::mat x = ones(size(y))%(y>zeros(size(y))); return(x);}
arma::mat delu(    const arma::mat& y); //define below
arma::mat dlrelu(  const arma::mat& y); 
arma::mat dtanh(   const arma::mat& y) {arma::mat x = tanh(y); return (1.0-x%x);}  // sech^2(x) = 1 - tanh^2(x)

arma::mat  actfun(const arma::mat& x, const int& i);
arma::mat dactfun(const arma::mat& y, const int& i);

List getCost(const arma::mat& y, const arma::mat& yh, const int& i, const arma::mat& sample_weight);

//// main functions;
//List fwdNN2(const arma::mat& X, const List& model);
//List bwdNN2(const arma::mat& dy, List cache, const List& model);
//List optimizerMomentum(List V, List dW, List W, double alpha, double lr, double lambda);
