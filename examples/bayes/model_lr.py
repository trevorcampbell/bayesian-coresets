import numpy as np

def load_data(dnm):
  data = np.load(folder+'/'+dnm+'.npz')
  Z = data['y'][:, np.newaxis]*data['X']
  Zt = data['yt'][:, np.newaxis]*data['X']
  data.close()
  return Z, Zt

def gen_synthetic(n):
  mu = np.array([0, 0])
  cov = np.eye(2)
  th = np.array([3, 3])
  X = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y =(np.random.rand(n) <= ps).astype(int)
  return y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_joint(z, th, wts):
  return (wts*(-np.log1p(np.exp(-(z*th).sum(axis=1))))).sum() -(th**2).sum()/2.

def log_likelihood(z, th):
  if len(z.shape) == 1:
    return -np.log1p(np.exp(-(th*z).sum()))
  else:
    return -np.log1p(np.exp(-(th*z).sum(axis=1)))

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(z, th):
  if len(z.shape) == 1:
    es = np.exp(-(th*z).sum())
    return es/(1.+es)*z
  else:
    es = np.exp(-(th*z).sum(axis=1))
    return (es/(1.+es))[:, np.newaxis]*z

def grad_log_prior(th):
  return -th

def hess_log_joint(z, th):
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


