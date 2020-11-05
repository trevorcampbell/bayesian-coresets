import numpy as np
from scipy.special import gammaln

def load_data(dnm):
    """
    Loadsolve

    Args:
        dnm: (str): write your description
    """
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  Z = np.hstack((X, data['y'][:, np.newaxis]))
  Zt = np.hstack((Xt, data['yt'][:, np.newaxis]))
  data.close()
  return X[:, :-1], Y, Z, Zt, Z.shape[1]-1

def gen_synthetic(n):
    """
    Generate random samples.

    Args:
        n: (array): write your description
    """
  X = np.random.randn(n)
  X = np.hstack((X[:, np.newaxis], np.ones(X.shape[0])[:, np.newaxis]))
  y = np.random.poisson(np.log1p(np.exp(X.dot(np.array([1., 0.])))))
  return np.hstack((X, y[:, np.newaxis])), np.linalg.inv((X.T).dot(X)).dot((X.T).dot(y))

def compute_s(th, x):
    """
    Compute the s

    Args:
        th: (str): write your description
        x: (dict): write your description
    """
  s = x.dot(th.T)
  idcs = s > -100
  s[idcs] =  np.log(np.maximum(s[idcs], 0) + np.log1p(np.exp(-np.fabs(s[idcs]))))
  #for any indices s < -100, then log( log(1+e^s) ) approximately s
  return s

def log_likelihood(z, th):
    """
    Computes the likelihood

    Args:
        z: (array): write your description
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = np.tile(z[:, -1][:, np.newaxis], (1, th.shape[0]))
  s = compute_s(th, x)
  return y*s - gammaln(y+1) - np.exp(s)

def log_prior(th):
    """
    Log prior.

    Args:
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  return -0.5*th.shape[1]*np.log(2.*np.pi) - 0.5*(th**2).sum(axis=1)

def log_joint(z, th, wts):
    """
    Log - likelihood.

    Args:
        z: (array): write your description
        th: (array): write your description
        wts: (array): write your description
    """
  return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0) + log_prior(th)

def grad_th_log_likelihood(z, th):
    """
    Compute the log likelihood of the log likelihood.

    Args:
        z: (array): write your description
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = np.tile(z[:, -1][:, np.newaxis], (1, th.shape[0]))
  s = compute_s(th, x)
  g = y - np.exp(s)
  idcs = np.exp(s) > 1e-15
  g[idcs] = (y[idcs]*np.exp(-s[idcs]) - 1.)*(1. - np.exp(-np.exp(s[idcs])))
  return g[:, :, np.newaxis]*x[:, np.newaxis, :]

def grad_z_log_likelihood(z, th):
    """
    Compute the gradient of - likelihood of the likelihood

    Args:
        z: (array): write your description
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = np.tile(z[:, -1][:, np.newaxis], (1, th.shape[0]))
  s = compute_s(th, x)
  g = y - np.exp(s)
  idcs = np.exp(s) > 1e-15
  g[idcs] = (y[idcs]*np.exp(-s[idcs]) - 1.)*(1. - np.exp(-np.exp(s[idcs])))
  return g[:, :, np.newaxis]*th[:, np.newaxis, :]

def grad_th_log_prior(th):
    """
    Gradient prior

    Args:
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  return -th

def grad_th_log_joint(z, th, wts):
    """
    Gradient log - likelihood.

    Args:
        z: (todo): write your description
        th: (todo): write your description
        wts: (todo): write your description
    """
  return grad_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis]*grad_th_log_likelihood(z, th)).sum(axis=0)

def hess_th_log_likelihood(z, th):
    """
    Compute the log likelihood of the log likelihood.

    Args:
        z: (array): write your description
        th: (array): write your description
    """
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  x = z[:, :-1]
  y = np.tile(z[:, -1][:, np.newaxis], (1, th.shape[0]))
  s = compute_s(th, x)
  h = -(1.+y)*np.exp(s)
  idcs = np.exp(s) > 1e-15
  h[idcs] = (y[idcs]*np.exp(-s[idcs])*(1.-np.exp(-s[idcs]+np.exp(s[idcs]))+np.exp(-s[idcs])) - 1.)*(np.exp(-np.exp(s[idcs]))-np.exp(-2*np.exp(s[idcs])))
  return h[:, :, np.newaxis, np.newaxis]*x[:, np.newaxis, :, np.newaxis]*x[:, np.newaxis, np.newaxis, :]

def hess_th_log_prior(th):
    """
    Computes the probability of the prior.

    Args:
        th: (array): write your description
    """
  th = np.atleast_2d(th)
  return np.tile(-np.eye(th.shape[1]), (th.shape[0], 1, 1)) 

def hess_th_log_joint(z, th, wts):
    """
    Evaluates the log - likelihood.

    Args:
        z: (todo): write your description
        th: (todo): write your description
        wts: (todo): write your description
    """
  return hess_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis, np.newaxis]*hess_th_log_likelihood(z, th)).sum(axis=0)
  
#this line follows the format "model title, model code"  
stan_representation = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0> y[n]; // outputs
  matrix[n,d] x; // inputs
}
parameters {
  vector[d] theta; // auxiliary parameter
}
transformed parameters {
  vector[n] f;
  f = -log_inv_logit(-(x*theta));
}
model {
  theta ~ normal(0, 1);
  y ~ poisson(f);
}
"""

