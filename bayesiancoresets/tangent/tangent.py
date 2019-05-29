import numpy as np
import warnings
from .. import TOL

class TangentSpace(object):
  #keep track of a set of idcs (the tangent space subsample) and an optional "large" set of indices for more accurate vector sum computations
  def __init__(self, w, d):
    self.alg_name = self.__class__.__name__
    self.refw = w
    self.d = d
      
  #return the tangent vector for datapoint k (or slice)
  def __getitem__(self, k):
    if isinstance(k, (np.integer)):
      self._getslice([k,:])
    elif isinstance(k, slice):
      self._getslice(k)
    else:
      raise KeyError

  def update(self, w=None, d=None):
    #TODO check if d is none/ w is none
    if w != self.refw:
      self.d = d
      self.refw = w
      self._w_and_dim_changed()
    elif d != self.d:
      self.d = d
      self._dim_changed()
     
  #methods to be implemented
  def _getslice(self, k):
    raise NotImplementedError

  def _w_and_dim_changed(self):
    raise NotImplementedError

  def _dim_changed(self):
    raise NotImplementedError

  def sum(self):
    raise NotImplementedError
  
  def sum_w(self, w):
    raise NotImplementedError

  def residual(self, w):
    return self.sum_w(w) - self.sum()


#store fixed vectors, init takes vectors, get/set just returns np slices
#warnings if many are 0 vectors (< TOL)
#update dimension as an unimplemented method
class ProjectedTangentSpace(TangentSpace):
  def __init__(self, vecs):
    super().__init__()
    if len(vecs.shape) != 2:
      raise ValueError(self.alg_name+'.__init__(): vecs must be a 2d array, otherwise the expected behaviour is ambiguous')
    self.vecs = vecs
    self.vsum = vecs.sum(axis=0)
    if ( np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum() > self.vecs.shape[0]*0.25:
      warnings.warn(self.alg_name+'.__init__(): more than 25% of the vectors have norm less than TOL. # = ' + str(np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum())

  def _getslice(self, k):
    return vecs[k]

  def sum(self):
    return self.vsum
  
  def sum_w(self, w):
    return w.dot(vecs[:w.shape[0], :])

  def update_dimension(self):
    raise NotImplementedError

#run random feature projection to start, call parent init, then get/set
#update dim via projection code below
class RandomProjectedTangentSpace(ProjectedTangentSpace):
  def __init__(self, ):
    pass

#rather than random sampling for projection, do something smarter...
class OptimizedProjectedTangentSpace(ProjectedTangentSpace):
  def __init__(self):
    raise NotImplementedError

#noisy estimates of vectors, new random proj each time (avoids fixed error from above proj)
#update dim just sets a fixed member d that tells random proj how many components to sample
class MonteCarloTangentSpace(TangentSpace):
  def __init__(self):
    pass


import numpy as np

class Projection(object):

  def __init__(self, data, log_likelihood, projection_dim, sample_approx_posterior):
    self.data = data
    self.log_likelihood = log_likelihood
    _Projection.__init__(self, data.shape[0], projection_dim, sample_approx_posterior)

  def _sample_component(self):
    return self.log_likelihood(self.data, self.sample_approx_posterior())


  def __init__(self, N, projection_dim, sample_approx_posterior):
    self.dim = sample_approx_posterior().shape[0]
    self.x = np.zeros((N, 0))
    self.sample_approx_posterior = sample_approx_posterior
    self.update_dimension(projection_dim)
    return

  def update_dimension(self, projection_dim):
    if projection_dim < self.x.shape[1]:
      self.x = self.x[:, :projection_dim]

    if projection_dim > self.x.shape[1]:
      old_dim = self.x.shape[1]
      w = np.zeros((self.x.shape[0], projection_dim))
      w[:, :old_dim] = self.x
      w *= np.sqrt(old_dim)
      for j in range(projection_dim-old_dim):
          w[:, j+old_dim] = self._sample_component()
      w /= np.sqrt(projection_dim)
      self.x = w
    return

  def reset(self, projection_dim=None):
    if projection_dim is None:
      projection_dim = self.x.shape[1]
    self.update_dimension(0)
    self.update_dimension(projection_dim)
    return

  def get(self):
    return self.x.copy()
  
