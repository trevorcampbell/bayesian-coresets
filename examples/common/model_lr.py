import numpy as np

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  Z = data['y'][:, np.newaxis]*X
  Zt = data['yt'][:, np.newaxis]*Xt
  data.close()
  return Z, Zt, Z.shape[1]

def gen_synthetic(n):
  mu = np.array([0, 0])
  cov = np.eye(2)
  th = np.array([3, 3])
  X = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y = (np.random.rand(n) <= ps).astype(int)
  y[y==0] = -1
  return y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

def log_likelihood_2d2d(z, th):
  lls = np.zeros((z.shape[0], th.shape[0]))
  for i in range(th.shape[0]):
    lls[:, i] = log_likelihood(z, th[i,:])
  return lls

def log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = -np.log1p(np.exp(m))
    else:
      m = -m
    return m 
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = -np.log1p(np.exp(m[idcs]))
    m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
    return m

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(z, th, idx=None):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = np.exp(m)/(1.+np.exp(m))
    else:
      m = 1.
    return m*z
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
    m[np.logical_not(idcs)] = 1.
    if idx is None:
      return m[:, np.newaxis]*z
    return m*z[:, idx]

def grad_log_prior(th):
  return -th

def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

def hess_log_joint(z, th):
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior

def hess_log_joint_w(z, th, wts):
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(wts*z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


