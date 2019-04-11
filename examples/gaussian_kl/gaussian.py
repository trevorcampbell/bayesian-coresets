import autograd.numpy as np
from autograd import grad
import warnings
 
def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, Siginv, x, w): 
  Sigp = np.linalg.inv(Sig0inv + w.sum()*Siginv)
  mup = np.dot(Sigp,  np.dot(Sig0inv,th0) + np.dot(Siginv, (w[:, np.newaxis]*x).sum(axis=0)))
  return mup, Sigp

def weighted_post_KL(th0, Sig0inv, Siginv, x, w, reverse=True):
  muw, Sigw = weighted_post(th0, Sig0inv, Siginv, x, w)
  mup, Sigp = weighted_post(th0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  if reverse:
    return gaussian_KL(muw, Sigw, mup, np.linalg.inv(Sigp))
  else:
    return gaussian_KL(mup, Sigp, muw, np.linalg.inv(Sigw))

#NB: without constant terms
#E[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m1_exact(muw, Sigw, Siginv, x):
  return -0.5*np.trace(np.dot(Siginv, Sigw)) -0.5*(np.dot((x - muw), Siginv)*(x-muw)).sum(axis=1)

#Covar[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq*musq[:, np.newaxis] + np.diag(Rho).sum()*(musq + musq[:,np.newaxis]) + 4*np.dot(np.dot(mu, Rho), mu.T))

#Var[Log N(x;, mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact_diag(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq**2 + 2*np.diag(Rho).sum()*musq + 4*(np.dot(mu, Rho)*mu).sum(axis=1))
 

