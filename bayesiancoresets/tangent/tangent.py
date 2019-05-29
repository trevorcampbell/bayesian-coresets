import numpy as np
import warnings
from .. import TOL

class TangentSpace(object):
  def __init__(self, d):
    self.alg_name = self.__class__.__name__
    self.d = d
      
  #return the tangent vector for datapoint k (or slice)
  def __getitem__(self, k):
    if isinstance(k, (np.integer)):
      self._getslice([k,:])
    elif isinstance(k, slice):
      self._getslice(k)
    else:
      raise KeyError

  #update the tangent space dimension
  def change_dimension(self, d):
    self.d = d
    self._dim_changed()

  def refresh(self, dim=None):
    dim = dim if dim is not None else self.d
    self.change_dimension(0)
    self.change_dimension(dim)
    return

  #methods to be implemented
  def _getslice(self, k):
    raise NotImplementedError

  def _dim_changed(self):
    raise NotImplementedError

  def sum(self):
    raise NotImplementedError
  
  def sum_w(self, w):
    raise NotImplementedError

  def residual(self, w):
    return self.sum_w(w) - self.sum()


class TangentSpaceProjection(TangentSpace):
  def _set_vecs(self, vecs):
    if len(vecs.shape) != 2:
      raise ValueError(self.alg_name+'._set_vecs(): vecs must be a 2d array, otherwise the expected behaviour is ambiguous')
    self.vecs = vecs
    self.vsum = vecs.sum(axis=0)
    if ( np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum() > self.vecs.shape[0]*0.25:
      warnings.warn(self.alg_name+'.__init__(): more than 25% of the vectors have norm less than TOL. # = ' + str(np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum())

  def _getslice(self, k):
    return vecs[k, :]

  def sum(self):
    return self.vsum
  
  def sum_w(self, w):
    return w.dot(vecs[:w.shape[0], :])

class MonteCarloTangentSpaceProjection(TangentSpaceProjection):
  def __init__(self, log_likelihood, sampler, d):
    super().__init__(d)
    self.log_likelihood = log_likelihood
    self.sampler = sampler
    self._set_vecs(self.log_likelihood(sampler.sample(1)))
    self._dim_changed()

  def _dim_changed(self):
    if self.d < self.vecs.shape[1]:
      self._set_vecs(self.vecs[:, :self.d])
    elif self.d > self.vecs.shape[1]:
      old_dim = self.vecs.shape[1]
      v = np.zeros((self.vecs.shape[0], self.d))
      v[:, :old_dim] = self.vecs
      v *= np.sqrt(old_dim)
      v[:, old_dim:] = self.log_likelihood(sampler.sample(self.d-old_dim))
      v /= np.sqrt(self.d)
      self._set_vecs(v)
    return

#rather than random sampling for projection, do something smarter...
class OptimizedTangentSpaceProjection(TangentSpaceProjection):
  def __init__(self):
    raise NotImplementedError

#noisy estimates of vectors, new random proj each time (avoids fixed error from above proj)
#update dim just sets a fixed member d that tells random proj how many components to sample
class MonteCarloTangentSpace(TangentSpace):
  def __init__(self, log_likelihood, sampler, d):
    super().__init__(d)
    self.log_likelihood = log_likelihood
    self.sampler = sampler
    self._dim_changed()

  def _dim_changed(self):
    pass #do nothing in the MC tangent space, since vecs are always created fresh

  def _getslice(self, k):
    return self.log_likelihood(self.sampler(self.d), idcs=k)

  def sum(self):
    return self.log_likelihood(self.sampler(self.d)).sum(axis=0)
  
  def sum_w(self, w):
    return w.dot(self.log_likelihood(self.sampler(self.d)))
