import numpy as np
import ..utils.warn
from .. import TOL

#TODO implement result caching on sumw
class TangentSpace(object):
  def __init__(self, d):
    self.alg_name = self.__class__.__name__
    self.d = d
      
  #return the tangent vector for datapoint k (or slice)
  def __getitem__(self, k):
    raise NotImplementedError

  #update the tangent space dimension
  def change_dimension(self, d):
    self.d = d
    self._dim_changed()

  def refresh(self, dim=None):
    dim = dim if dim is not None else self.d
    self.change_dimension(0)
    self.change_dimension(dim)
    return

  def num_vectors(self):
    raise NotImplementedError

  def _dim_changed(self):
    raise NotImplementedError

  def sum(self):
    raise NotImplementedError
  
  def sum_w(self, w, idcs):
    raise NotImplementedError

  def norms(self):
    raise NotImplementedError

  def norms_sum(self):
    raise NotImplementedError

  def sum_norm(self):
    raise NotImplementedError

  def sum_w_norm(self):
    raise NotImplementedError

  def residual(self, w, idcs):
    return self.sum() - self.sum_w(w, idcs)

  def error(self, w, idcs):
    return np.sqrt((self.residual(w, idcs)**2).sum())

  def optimal_scaling(self, w, idcs):
    xw = self.sum_w(w, idcs)
    xwn = np.sqrt((xw**2).sum())
    xs = self.sum()
    xsn = np.sqrt((xs**2).sum())
    if xwn == 0. or xsn == 0.:
      return 0.
    if xwn < TOL or xsn < TOL:
        warn.warn(self.alg_name+'._optimal_scaling(): the norm of xs or xw is small; optimal scaling might be unstable. ||xs|| = ' + str(xsn) + ' ||xw|| = ' + str(xwn))
    return xsn/xwn*max(0., (xw/xwn).dot(xs/xsn))


class FiniteTangentSpace(TangentSpace):
  def _set_vecs(self, vecs):
    if len(vecs.shape) != 2:
      raise ValueError(self.alg_name+'._set_vecs(): vecs must be a 2d array, otherwise the expected behaviour is ambiguous')
    if vecs.shape[1] != self.d:
      raise ValueError(self.alg_name+'._set_vecs(): vecs must have the correct dimension')
    self.vecs = vecs
    self.vsum = vecs.sum(axis=0)
    self.vsum_norm = np.sqrt((self.vsum**2).sum())
    self.vnorms = np.sqrt((self.vecs**2).sum(axis=1))
    self.vnorms_sum = self.vnorms.sum()
    if ( np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum() > self.vecs.shape[0]*0.25:
      warn.warn(self.alg_name+'.__init__(): more than 25% of the vectors have norm less than TOL. # = ' + str(np.sqrt((self.vecs**2).sum(axis=1)) < TOL).sum())

  def __getitem__(self, k):
    return self.vecs[k]

  def sum(self):
    return self.vsum
  
  def sum_w(self, w, idcs):
    return w.dot(self.vecs[idcs,:])

  def sum_w_norm(self, w, idcs):
    return np.sqrt(((w.dot(self.vecs[idcs,:]))**2).sum())

  def num_vectors(self):
    return self.vecs.shape[0]

  def norms(self):
    return self.vnorms
 
  def norms_sum(self):
    return self.vnorms_sum

  def sum_norm(self):
    return self.vsum_norm

class FixedFiniteTangentSpace(FiniteTangentSpace):
  def __init__(self, vecs):
    super().__init__(vecs.shape[1])
    self._set_vecs(vecs)

  def _dim_changed(self):
    raise NotImplementedError(self.alg_name+'._dim_changed(): Cannot change the dimension of a fixed tangent space')


class MonteCarloFiniteTangentSpace(FiniteTangentSpace):
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

##TODO
##use compression (SVD, GIGA) to shrink an oversampled space
#class CompressedFiniteTangentSpace(TangentSpaceProjection):
#  def __init__(self):
#    raise NotImplementedError
#

##TODO
##rather than random sampling for projection, do something smarter...
#class OptimizedFiniteTangentSpace(TangentSpaceProjection):
#  def __init__(self):
#    raise NotImplementedError
#
#
##noisy estimates of vectors, new random proj each time (avoids fixed error from above proj)
##update dim just sets a fixed member d that tells random proj how many components to sample
##TODO implement new funcs above
#class MonteCarloTangentSpace(TangentSpace):
#  def __init__(self, log_likelihood, sampler, d):
#    super().__init__(d)
#    self.log_likelihood = log_likelihood
#    self.sampler = sampler
#    self._dim_changed()
#
#  def _dim_changed(self):
#    pass #do nothing in the MC tangent space, since vecs are always created fresh
#
#  def _getslice(self, k):
#    return self.log_likelihood(self.sampler(self.d), idcs=k)
#
#  def sum(self):
#    return self.log_likelihood(self.sampler(self.d)).sum(axis=0)
#  
#  def sum_w(self, w):
#    return w.dot(self.log_likelihood(self.sampler(self.d)))
