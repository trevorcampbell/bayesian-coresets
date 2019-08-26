import numpy as np


def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post_KL(th0, Sig0inv, sigsq, X, Y, w, reverse=True):
  muw, Sigw = weighted_post(th0, Sig0inv, sigsq, X, Y, w)
  mup, Sigp = weighted_post(th0, Sig0inv, sigsq, X, Y, np.ones(X.shape[0]))
  if reverse:
    return gaussian_KL(muw, Sigw, mup, np.linalg.inv(Sigp))
  else:
    return gaussian_KL(mup, Sigp, muw, np.linalg.inv(Sigw))

def weighted_post(th0, Sig0inv, sigsq, X, Y, w): 
  Sigp = np.linalg.inv(Sig0inv + (w[:, np.newaxis]*X).T.dot(X)/sigsq)
  mup = np.dot(Sigp,  np.dot(Sig0inv,th0) + (w[:, np.newaxis]*Y[:,np.newaxis]*X).sum(axis=0)/sigsq )
  return mup, Sigp

def potentials(sigsq, X, Y, samples):
  XST = X.dot(samples.T)
  return -1./2.*np.log(2.*np.pi*sigsq) - 1./(2.*sigsq)*(Y[:,np.newaxis]**2 - 2*XST*Y[:,np.newaxis] + XST**2)
 
