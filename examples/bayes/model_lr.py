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
  y =(np.random.rand(n) <= ps).astype(int)
  return y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

def log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = -np.log1p(np.exp(m))
    else:
      m = -m
    return m 
    #-np.log1p(np.exp(-(th*z).sum()))
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = -np.log1p(np.exp(m[idcs]))
    m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
    return m
    #return -np.log1p(np.exp(-(th*z).sum(axis=1)))

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = np.exp(m)/(1.+np.exp(m))
    else:
      m = 1.
    return m*z
    #es = np.exp(-(th*z).sum())
    #return es/(1.+es)*z
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
    m[np.logical_not(idcs)] = 1.
    return m[:, np.newaxis]*z
    #es = np.exp(-(th*z).sum(axis=1))
    #return (es/(1.+es))[:, np.newaxis]*z

def grad_log_prior(th):
  return -th

def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

def hess_log_joint(z, th):
  #m = -(th*z).sum(axis=1)
  #idcs = m < 100
  #m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  #m[np.logical_not(idcs)] = 1.
  #H_log_like = -(z.T).dot((m**2)[:, np.newaxis]*z)
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


