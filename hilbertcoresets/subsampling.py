import numpy as np
import warnings
from geometry import *

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
    # if norm sum > 0, use IS. Otherwise, all vecs are 0 and run() just outputs 0, so no need for ps
    if self.sig > 0.:
      self.ps = self.norms/self.sig
    else:
      self.ps = None
    self.xs = self.x.sum(axis=0)
    self.f_preproc = x.shape[0] + 2*self.x.shape[0] 
    self.diam = None
    self.normratio = None
    self.reset()

  def run(self, M):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('ImportanceSampling.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0:
      warnings.warn('ImportanceSampling.run(): data has no nonzero vectors. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.cts/self.ps/M
    self.M = M
    return

  def get_num_ops(self):
    return self.f_preproc

  def get_num_nodes(self):
    return 0.

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
    #if no nonzero vecs, just output 0
    if self.x.size == 0:
      return 0.
    #check validity of M
    M = np.floor(M) if M else self.M
    if M <= 0:
      raise ValueError('ImportanceSampling.sqrt_bound(): M must be >= 1. Requested M: '+str(M))
    #if the probability of error requested is 0, output a bound of inf
    if delta == 0.:
      return np.inf
    #compute diam/normratio if required
    if not self.diam:
      self.diam = compute_diam(self.x)
    if not self.normratio:
      self.normratio = compute_normratio(self.x)
    #if either normratio or diam is 0, output 0
    if self.diam == 0. or self.normratio == 0.:
      return 0.
    else:
      v = np.sqrt(2.*M*self.normratio**2/(self.diam**2*np.log(1./delta)))
      nm = min(self.diam, self.normratio*v*self._hinv(1./v**2))
    return self.sig/np.sqrt(M)*(self.normratio + nm*np.sqrt(2.*np.log(1./delta)))

  def _hinv(self, v):
    #search for bounding interval
    yL = 0.
    yR = 1.
    while (1.+yR)*np.log(1.+yR)-yR < v:
      yR *= 2
    #bisection search on the monotone function for inverse
    while (yR-yL)/yR > 1e-12:
      yC = (yL+yR)/2.
      vC = (1.+yC)*np.log(1.+yC)-yC
      if vC > v:
        yR = yC
      else:
        yL = yC
    return (yL+yR)/2.

class RandomSubsampling(ImportanceSampling):
  def __init__(self, x):
    ImportanceSampling.__init__(self, x)
    if self.N > 0: #otherwise self.ps = None from the ImportanceSampling constructor above
      self.ps = 1.0/float(self.N) * np.ones(self.N)
    #self.xi = None
    #self.tau = None

  def sqrt_bound(self, delta, M=None):
    raise NotImplementedError("RandomSubsampling.sqrt_bound():  not implemented")
    #if self.x.size == 0:
    #  return 0.
    #if delta == 0.:
    #  return np.inf
    #if not self.xi or not self.tau:
    #  self._compute_xi_tau()
    #M = M if M else self.M
    #if self.xi == 0. or self.tau == 0.:
    #  nm = 0.
    #else:
    #  v = self.tau*np.sqrt(M)/self.xi/np.sqrt(np.log(1./delta))
    #  nm = min(np.sqrt(2)*self.xi/self.tau, v*self._hinv(1./v**2))
    #return np.sqrt(self.tau/float(M))*( np.sqrt(1./2.) + nm*np.sqrt(self.tau*np.log(1./delta)))

  
