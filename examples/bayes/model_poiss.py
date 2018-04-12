import numpy as np
from scipy.special import gammaln

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  Z = np.hstack((X, data['y'][:, np.newaxis]))
  Zt = np.hstack((Xt, data['yt'][:, np.newaxis]))
  data.close()
  return Z, Zt, Z.shape[1]-1

def gen_synthetic(n):
  X = np.random.randn(n)
  X = np.hstack((X[:, np.newaxis], np.ones(X.shape[0])[:, np.newaxis]))
  y = np.random.poisson(np.log1p(np.exp(X.dot(np.array([1., 0.])))))
  return np.hstack((X, y[:, np.newaxis])), np.linalg.inv((X.T).dot(X)).dot((X.T).dot(y))

def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

def compute_m(th, x):
  m = 0.
  if len(x.shape) == 1:
    m = (th*x).sum()
    if m < 100:
      m = np.log1p(np.exp(m))
  else:
    m = (th*x).sum(axis=1)
    m[m<100] = np.log1p(np.exp(m[m<100]))
  return m

def log_likelihood(Z, th):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)
  return y*np.log(m) - gammaln(y+1) - m

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(Z, th):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)
  return ((1.-y/m)*np.expm1(-m))[:, np.newaxis]*x

def grad_log_prior(th):
  return -th

def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

def hess_log_joint(Z, th):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)
  H_log_like = (np.exp(np.log((np.expm1(m) - m)*y + m**2) - m - 2*np.log(m))*np.expm1(-m)*x.T).dot(x)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior



