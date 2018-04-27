import numpy as np
import warnings

class CoresetConstruction(object):
  def __init__(self, _x):
    self.alg_name = self.__class__.__name__
    #convert x to a 2d N x D numpy array with each row = D-dim data vector
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError(self.alg_name + ': input must be a 2d numeric ndarray')
    #extract data with nonzero norm, save original size and nonzero index locations
    self.full_N = x.shape[0]
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    #compute the sum vector
    self.xs = x[self.nzidcs, :].sum(axis=0)
    self.snorm = np.sqrt((self.xs**2).sum())
    #normalize the sum vector if xs != 0; otherwise just leave it alone (algorithms will just output wts = 0)
    if self.snorm > 0: 
      self.xs /= self.snorm
    #save norms / data / size for nonzero vectors
    self.norms = nrms[self.nzidcs]
    self.x = x[self.nzidcs, :]/self.norms[:, np.newaxis]
    self.norm_sum = self.norms.sum()
    self.N = self.x.shape[0]
    #call reset to initialize weights, weighted sum; stored quantities that will be updated 
    self.reset()

  def reset(self):
    self.M = 0
    self.coreset_size = 0
    self.reached_numeric_limit = False
    self.wts = np.zeros(self.N)
    self.xw = np.zeros(self.x.shape[1])

  def coreset_size(self):
    return (self.wts > 0).sum()

  def weights(self, optimal_scaling=False, use_cached_xw=False):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    #make sure the weights apply to the original unnormalized data
    full_wts[self.nzidcs] = self.wts/self.norms
    #if xw is not scaled properly (e.g. normalized, as in GIGA) or if the user explicitly asks for it, optimally scale
    if self._xw_unscaled() or optimal_scaling:
      return full_wts*self._optimal_scaling(self.xw if use_cached_xw else self.wts.dot(self.x))
    else:
      return full_wts

  def error(self, optimal_scaling=False, use_cached_xw=True):
    if use_cached_xw:
      yw = self.xw
    else:
      yw = self.wts.dot(self.x)

    if self._xw_unscaled() or optimal_scaling:
      yw *= self._optimal_scaling(yw)

    return np.sqrt(((yw-self.snorm*self.xs)**2).sum())

  def run(self, M, use_cached_xw=True):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError(self.alg_name+'.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0:
      warnings.warn(self.alg_name+'.run(): data has no nonzero vectors. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return
    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.run(): the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    #initialize optimization
    if self.M == 0:
      self._initialize()
    
    #iterative construction
    for m in range(self.M, M):
      self._step(use_cached_xw)
      if self.reached_numeric_limit:
        break
      self.M = m+1

    return

  def _optimal_scaling(self, y):
    yn = np.sqrt((y**2).sum())
    return self.snorm/yn*max(0., (y/yn).dot(self.xs))

  def _step(self, use_cached_xw):
    raise NotImplementedError()

  def _initialize(self):
    raise NotImplementedError()

  def _xw_unscaled(self):
    raise NotImplementedError()
