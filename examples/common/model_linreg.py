import numpy as np
import scipy.linalg as sl

def log_likelihood(z, th, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  XST = x.dot(th.T)
  return -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def grad_x_log_likelihood(z, th, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  return 1./sigsq*(y[:, np.newaxis] - x.dot(th.T))[:,:,np.newaxis]*np.hstack((th, np.ones(th.shape[0])[:,np.newaxis]))[np.newaxis, :, :]

def KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, sigsq, z, w): 
  if w.shape[0] > 0:
      z = np.atleast_2d(z)
      X = z[:, :-1]
      Y = z[:, -1]
      LSigpInv = np.linalg.cholesky(Sig0inv + (w[:, np.newaxis]*X).T.dot(X)/sigsq) #SigpInv = LL^T, L lower tri
      USigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b = True, check_finite = False).T #Sigp = UU^T, U upper tri
      mup = np.dot(USigp.dot(USigp.T),  np.dot(Sig0inv,th0) + (w[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/sigsq )
  else:
      mup = th0
      LSigpInv = np.linalg.cholesky(Sig0inv)
      USigp = sl.solve_triangular(LSigpInv, np.eye(LSigpInv.shape[0]), lower=True, overwrite_b = True, check_finite = False).T
  return mup, USigp, LSigpInv


