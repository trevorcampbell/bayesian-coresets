import numpy as np

class ImportanceSampling(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('ImportanceSampling: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.N = self.x.shape[0]
    self.norms = nrms[self.nzidcs]
    self.sig = self.norms.sum()
    if self.sig > 0.:
      self.ps = self.norms/self.sig
    elif self.N > 0:
      self.ps = 1.0/float(self.N) * np.ones(self.N) 
    else:
      self.ps = None
    self.xs = x.sum(axis=0)
    self.diam = None
    self.normratio = None
    self.reset()

  def run(self, M):
    if self.x.size == 0:
      return
    if M <= self.M:
      print 'Warning: requested M <= self.M, returning without modifying weights'
      return
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.cts/self.ps/M
    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.cts = np.zeros(self.N)

  def weights(self):
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    return full_wts

  def error(self):
    return np.sqrt((((self.wts[:, np.newaxis]*self.x).sum(axis=0) - self.xs)**2).sum())

  def sqrt_bound(self, delta, M=None):
    if self.x.size == 0:
      return 0.
    if not self.diam:
      self._compute_diam()
    if not self.normratio:
      self._compute_normratio()
    M = M if M else self.M
    v = np.sqrt(2.*M*self.normratio**2/(self.diam**2*np.log(1./delta)))
    nm = min(self.diam, self.normratio*v*self._hinv(1./v**2))
    return self.sig/np.sqrt(M)*(self.normratio + nm*np.sqrt(2.*np.log(1./delta)))

  def _hinv(self, v):
    yL = 0.
    yR = 1.
    while (1.+yR)*np.log(1.+yR)-yR < v:
      yR *= 2
    while (yR-yL)/yR > 1e-12:
      yC = (yL+yR)/2.
      vC = (1.+yC)*np.log(1.+yC)-yC
      if vC > v:
        yR = yC
      else:
        yL = yC
    return (yL+yR)/2.

  def _compute_diam(self):
    normed_x = self.x/self.norms[:, np.newaxis] 
    distsqs = 2. - 2.*normed_x.dot(normed_x.T)
    self.diam = np.sqrt(distsqs.max())
    return

  def _compute_normratio(self):
    self.normratio = np.sqrt(1. - (self.xs**2).sum()/self.sig**2)
    return

class RandomSubsampling(ImportanceSampling):
  def __init__(self, x):
    ImportanceSampling.__init__(self, x)
    self.ps = 1.0/float(self.N) * np.ones(self.N)
    self.xi = None
    self.tau = None

  def sqrt_bound(self, delta, M=None):
    if not self.xi or not self.tau:
      self._compute_xi_tau()
    M = M if M else self.M
    v = self.tau*np.sqrt(M)/self.xi/np.sqrt(np.log(1./delta))
    nm = min(np.sqrt(2)*self.xi/self.tau, v*self._hinv(1./v**2))
    return np.sqrt(self.tau/float(M))*( np.sqrt(1./2.) + nm*np.sqrt(self.tau*np.log(1./delta)))

  def _compute_xi_tau(self):
    xnrmsqs = (self.x**2).sum(axis=1)
    distsqs = xnrmsqs[:, np.newaxis] + xnrmsqs - 2.*self.x.dot(self.x.T)
    self.xi = self.x.shape[0]*np.sqrt(distsqs.max())
    self.tau = distsqs.sum()
    return

