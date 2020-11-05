import numpy as np
import scipy.linalg as sl

def log_likelihood(x, th, Siginv, logdetSig):
    """
    R log - likelihood.

    Args:
        x: (array): write your description
        th: (array): write your description
        Siginv: (array): write your description
        logdetSig: (array): write your description
    """
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  xSiginvx = (x*(x.dot(Siginv))).sum(axis=1)
  thSiginvth = (th*(th.dot(Siginv))).sum(axis=1)
  xSiginvth = x.dot(Siginv.dot(th.T))
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] + thSiginvth - 2*xSiginvth)

def grad_x_log_likelihood(x, th, Siginv):
    """
    Compute the log - likelihood of x.

    Args:
        x: (array): write your description
        th: (array): write your description
        Siginv: (array): write your description
    """
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  return th.dot(Siginv)[np.newaxis, :, :] - x.dot(Siginv)[:, np.newaxis, :]
 
def KL(mu0, Sig0, mu1, Sig1inv):
    """
    Compute the log - likelihood.

    Args:
        mu0: (int): write your description
        Sig0: (todo): write your description
        mu1: (int): write your description
        Sig1inv: (todo): write your description
    """
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, Siginv, x, w): 
    """
    Compute the post - weight weight.

    Args:
        th0: (array): write your description
        Sig0inv: (array): write your description
        Siginv: (array): write your description
        x: (array): write your description
        w: (array): write your description
    """
  LSigpInv = np.linalg.cholesky(Sig0inv + w.sum()*Siginv) # SigpInv = LL^T, L lower tri
  USigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b=True, check_finite=False).T # Sigp = UU^T, U upper tri
  if w.shape[0] > 0:
      mup = np.dot(USigp.dot(USigp.T),  np.dot(Sig0inv,th0) + np.dot(Siginv, (w[:, np.newaxis]*x).sum(axis=0)))
  else:
      mup = th0
  return mup, USigp, LSigpInv
  
