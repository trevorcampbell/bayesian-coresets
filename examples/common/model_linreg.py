import numpy as np


def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post_KL(th0, Sig0inv, sigsq, z, w, reverse=True):
  muw, Sigw = weighted_post(th0, Sig0inv, sigsq, z, w)
  mup, Sigp = weighted_post(th0, Sig0inv, sigsq, z, np.ones(z.shape[0]))
  if reverse:
    return gaussian_KL(muw, Sigw, mup, np.linalg.inv(Sigp))
  else:
    return gaussian_KL(mup, Sigp, muw, np.linalg.inv(Sigw))

def weighted_post(th0, Sig0inv, sigsq, z, w): 
  z = np.atleast_2d(z)
  X = z[:, :-1]
  Y = z[:, -1]
  Sigp = np.linalg.inv(Sig0inv + (w[:, np.newaxis]*X).T.dot(X)/sigsq)
  mup = np.dot(Sigp,  np.dot(Sig0inv,th0) + (w[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/sigsq )
  return mup, Sigp

def potentials(z, th, sigsq):
  z = np.atleast_2d(z)
  x = z[:, :-1]
  y = z[:, -1]
  th = np.atleast_2d(th)
  XST = x.dot(th.T)
  return -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

