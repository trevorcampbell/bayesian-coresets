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
  ll = log_likelihood(Z, th)
  if np.any(np.logical_and(np.isinf(ll), wts>0)):
    return -np.inf
  return (wts[np.isfinite(ll)]*ll[np.isfinite(ll)]).sum() + log_prior(th)

def compute_m(th, x):
  m = np.atleast_2d(th*x).sum(axis=1)
  return np.maximum(m, 0) + np.log1p(np.exp(-np.fabs(m)))

def log_likelihood_2d2d(z, th):
  lls = np.zeros((z.shape[0], th.shape[0]))
  for i in range(th.shape[0]):
    lls[:, i] = log_likelihood(z, th[i,:])
  return lls


def log_likelihood(Z, th):
  x = Z[:, :-1]
  y = Z[:, -1]
  dots = np.atleast_2d(th*x).sum(axis=1)
  m = np.maximum(dots, 0) + np.log1p(np.exp(-np.fabs(dots)))
  ll = np.zeros(dots.shape[0])
  ll[m>0] = y[m>0]*np.log(m[m>0]) - gammaln(y[m>0]+1)-m[m>0]
  ll[m==0] = y[m==0]*dots[m==0] - gammaln(y[m==0]+1)
  return ll

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(Z, th, idx=None):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)
  g = np.zeros(y.shape[0])
  g[:] = y
  mnz = m[m>1e-100]
  ynz = y[m>1e-100]
  g[m > 1e-100] = ((1.-ynz/mnz)*np.expm1(-mnz))
  if idx is None:
    return g[:, np.newaxis]*x
  return g*x[:, idx]

def grad_log_prior(th):
  return -th

def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

def hess_log_joint(Z, th):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)

  #todo: instead of +1e-100 use the stabilization from log_likelihood(z,th) (ll[m>0] and ll[m==0])
  m += 1e-100 #just to prevent zeros
  H_log_like = (np.exp(np.log((1-np.exp(-m)*(1+m))*y + np.exp(-m)*m**2) - 2*np.log(m))*np.expm1(-m)*x.T).dot(x)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior

def hess_log_joint_w(Z, th, wts):
  x = Z[:, :-1]
  y = Z[:, -1]
  m = compute_m(th, x)

  #todo: instead of +1e-100 use the stabilization from log_likelihood(z,th) (ll[m>0] and ll[m==0])
  m += 1e-100 #just to prevent zeros
  H_log_like = (wts*(np.exp(np.log((1-np.exp(-m)*(1+m))*y + np.exp(-m)*m**2) - 2*np.log(m))*np.expm1(-m)*x.T)).dot(x)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior



