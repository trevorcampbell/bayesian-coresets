import numpy as np

def gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples):
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] - 2.*np.dot(xSiginv, samples.T) + (np.dot(samples, Siginv)*samples).sum(axis=1))
 
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


