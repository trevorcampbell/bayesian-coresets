import numpy as np
import warnings
from .coreset import Coreset
from .iterative import SingleGreedyCoreset, IterativeCoreset

class VectorCoreset(Coreset):
  def __init__(self, _x, use_cached_xw=True):
    #convert x to a 2d N x D numpy array with each row = D-dim data vector
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError(self.alg_name + ': input must be a 2d numeric ndarray')
    
    #extract data with nonzero norm, save original size and nonzero index locations
    self.use_cached_xw = use_cached_xw
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
    if self.x.size == 0:
      warnings.warn(self.alg_name+'.__init__(): data has no nonzero vectors. ' + self.alg_name+'.run() will return immediately')
    super(VectorCoreset, self).__init__(self.x.shape[0])

  def reset(self):
    super(VectorCoreset, self).reset()
    #reset the cached xw too
    self.xw = np.zeros(self.x.shape[1])

  def weights(self, optimal_scaling=False):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    #make sure the weights apply to the original unnormalized data
    full_wts[self.nzidcs] = self.wts/self.norms
    #if xw is not scaled properly (e.g. normalized, as in GIGA) or if the user explicitly asks for it, optimally scale
    if self._xw_unscaled() or optimal_scaling:
      return full_wts*self._optimal_scaling(self.xw if self.use_cached_xw else self.wts.dot(self.x))
    else:
      return full_wts

  def error(self, optimal_scaling=False):
    if self.use_cached_xw:
      yw = self.xw.copy()
    else:
      yw = self.wts.dot(self.x)

    if self._xw_unscaled() or optimal_scaling:
      yw *= self._optimal_scaling(yw)

    return np.sqrt(((yw-self.snorm*self.xs)**2).sum())

  def _optimal_scaling(self, y):
    yn = np.sqrt((y**2).sum())
    if yn > 0.:
      if yn < 1e-9:
        warnings.warn(self.alg_name+'._optimal_scaling(): the norm of y is small; optimal scaling might be unstable. ||y|| = ' + str(yn))
      return self.snorm/yn*max(0., (y/yn).dot(self.xs))
    return 0.

  def _xw_unscaled(self):
    raise NotImplementedError()

class IterativeVectorCoreset(VectorCoreset, IterativeCoreset):
  pass

class SingleGreedyVectorCoreset(VectorCoreset, SingleGreedyCoreset):

  def _renormalize(self):
    nrm = np.sqrt((self.xw**2).sum())
    self.xw /= nrm
    self.wts /= nrm

  def _prepare_retry_step(self):
    #try recomputing xw from scratch and rerunning search
    self.xw = self.wts.dot(self.x)
    if self._xw_unscaled():
      self._renormalize()
  
  def _prepare_retry_search(self):
    self._prepare_retry_step()

  def _update_cache(self, alpha, beta, f):
    #apply the same update to xw
    if self.use_cached_xw:
      self.xw = alpha*self.xw + beta*self.x[f, :]
    else:
      self.xw = self.wts.dot(self.x)
    #if xw is unscaled, renormalize
    if self._xw_unscaled():
      self._renormalize()
   
