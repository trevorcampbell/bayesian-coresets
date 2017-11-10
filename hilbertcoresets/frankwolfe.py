import numpy as np
from geometry import *
import warnings

class FrankWolfe(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('FrankWolfe: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.norms = nrms[self.nzidcs]
    self.sig = self.norms.sum()
    self.xs = self.x.sum(axis=0)
    self.N = self.x.shape[0]
    self.normratio = None
    self.diam = None
    self.nu = None
    self.f_search = 0.
    self.f_update = 0.
    self.reached_numeric_limit = False
    self.reset()

  #options are fast, accurate (fast tracks xw and wts separately, accurate updates xw from wts at each iter)
  def run(self, M, update_method='fast'):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('FrankWolfe.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or self.reached_numeric_limit:
      warnings.warn('FrankWolfe.run(): either data has no nonzero vectors or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    #initialize optimization
    if self.M == 0:
      f = self.search()
      self.wts[f] = self.sig/self.norms[f]
      self.xw = self.sig/self.norms[f]*self.x[f, :]
      self.M = 1
      self.f_update += 1.

    for m in range(self.M, M):
      #search for FW vertex and compute line search
      f = self.search()
      gammanum = (self.sig/self.norms[f]*self.x[f, :] - self.xw).dot(self.xs - self.xw)
      gammadenom = ((self.sig/self.norms[f]*self.x[f, :] - self.xw)**2).sum()
      self.f_update += 4.
      
      #if the line search is invalid, possibly reached numeric limit
      #try recomputing xw from scratch and rerunning search
      if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
        self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)
        f = self.search()
        gammanum = (self.sig/self.norms[f]*self.x[f, :] - self.xw).dot(self.xs - self.xw)
        gammadenom = ((self.sig/self.norms[f]*self.x[f, :] - self.xw)**2).sum()
        self.f_update += 4. + (self.wts > 0).sum()
        #if it's still no good, we've reached the numeric limit
        if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:  
          self.reached_numeric_limit = True
          break
      #update xw, wts, M
      gamma = gammanum/gammadenom
      self.wts *= (1.-gamma)
      self.wts[f] += gamma*self.sig/self.norms[f] 
      if update_method == 'fast':
        self.xw = (1.-gamma)*self.xw + gamma*self.sig/self.norms[f]*self.x[f, :]
        self.f_update += 1.
      else:
        self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)
        self.f_update += (self.wts > 0).sum()
      self.M = m+1

    return

  def search(self):
    scores = ((self.xs - self.xw)*self.x).sum(axis=1)/self.norms
    self.f_search += 2. + self.x.shape[0]
    return scores.argmax()

  def get_num_ops(self):
    return self.f_search + self.f_update

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.xw = np.zeros(self.x.shape[1])
    self.reached_numeric_limit = False
    self.f_search = 0.
    self.f_update = 0.

  def weights(self):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    return full_wts

  #options are fast, accurate (either use xw or recompute xw from wts)
  def error(self, method="fast"):
    if method == "fast":
      return np.sqrt(((self.xw - self.xs)**2).sum())
    else:
      return np.sqrt((((self.wts[:, np.newaxis]*self.x).sum(axis=0) - self.xs)**2).sum())

  def exp_bound(self, M=None):
    #if no nonzero data, always return 0 since we output wts = 0
    if self.x.size == 0:
      return 0.
    #check M validity
    M = np.floor(M) if M else self.M
    if M <= 0:
      raise ValueError('FrankWolfe.exp_bound(): M must be >= 1. Requested M: '+str(M))
    #if the dimension is large, qhull may take a long time or fail
    if self.x.shape[1] > 3:
      warnings.warn('FrankWolfe.exp_bound(): this code uses scipy.spatial.ConvexHull (QHull) which may fail or run slowly for high dimensional data.')
    #compute diam and normratio if we need to 
    if not self.diam:
      self.diam = compute_diam(self.x)
    if not self.normratio:
      self.normratio = compute_normratio(self.x)
    #if the diam or normratio are 0, all points are aligned and err = 0
    if self.diam == 0. or self.normratio == 0.:
      return 0.
    #compute nu if necessary
    if not self.nu:
      self.nu, _ = compute_nu(self.x, self.diam)
    #if nu is 0: if M == 1, we get sig*normratio error bound; if M > 1, we get 0.
    if self.nu == 0.:
      return 0. if M > 1 else self.sig*self.normratio
    lognum = np.log(self.sig) + np.log(self.normratio) + np.log(self.diam) + np.log(self.nu)
    logdenom_a = 2*np.log(self.diam) - 2.*(M-2.)*np.log(self.nu)
    if M > 1:
      logdenom_b = 2*np.log(self.normratio) + np.log(M-1.)
    else:
      logdenom_b = -np.inf 
    logdenom = 0.5*np.logaddexp(logdenom_a, logdenom_b)
    return np.exp(lognum - logdenom)
  
  def sqrt_bound(self, M=None):
    #if no nonzero data, error always 0 since we output wts = 0
    if self.x.size == 0:
      return 0.
    #check M validity
    M = np.floor(M) if M else self.M
    if M <= 0:
      raise ValueError('FrankWolfe.exp_bound(): M must be >= 1. Requested M: '+str(M))
    #if diam not yet computed, compute it
    if not self.diam:
      self.diam = compute_diam(self.x)
    return self.sig*self.diam/np.sqrt(M)


