import numpy as np

#TODO: remove any zero weight obs and change weights() to remap

class ImportanceSampling(object):
  def __init__(self, x):
    self.N = x.shape[0]
    self.norms = np.sqrt((x**2).sum(axis=1))
    self.sig = self.norms.sum()
    if self.sig > 0.:
      self.ps = self.norms/self.sig
    else:
      print 'Warning: sum of norms is 0; falling back to random subsampling' 
      self.ps = 1.0/float(self.N) * np.ones(self.N) 
    self.ps = self.norms/self.sig
    self.x = x
    self.xs = x.sum(axis=0)
    self.diam = None
    self.normratio = None
    self.reset()

  def run(self, M):
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
    return self.wts

  def error(self):
    return np.sqrt((((self.wts[:, np.newaxis]*self.x).sum(axis=0) - self.xs)**2).sum())

  def sqrt_bound(self, delta, M=None):
    if not self.diam:
      self._compute_diam()
    if not self.normratio:
      self._compute_normratio()
    M = M if M else self.M
    v = np.sqrt(2.*M*self.normratio**2/(self.diam**2*np.log(1./delta)))
    nm = min(self.diam, self.normratio*v*self._hinv(1./v))
    return self.sig/np.sqrt(M)*(self.normratio + nm*np.sqrt(2.*np.log(1./delta)))

  def _hinv(v):
    yL = 0.
    yR = 1.
    while (1.+yR)*np.log(1.+yR)-yR < v:
      yR *= 2
    while (yR-yL)/yR > 1e-6:
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

  def sqrt_bound(self, delta, M=None):
    print 'Error: requested sqrt_bound of RandomSubsample. Not implemented, returning NaN.'
    return np.nan
