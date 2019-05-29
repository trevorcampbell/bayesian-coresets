

class TangentSpace(object):
  #return the tangent vector for datapoint k
  def __getitem__(self, k):
    raise NotImplementedError

  #set the tangent vector for datapoint k
  def __setitem__(self, k):
    raise NotImplementedError

#store fixed vectors, init takes vectors, get/set just returns np slices
#warnings if many are 0 vectors (< TOL)
#update dimension as an unimplemented method
class ProjectedTangentSpace(TangentSpace):
  def __init__(self):
    pass

#run random feature projection to start, call parent init, then get/set
#update dim via projection code below
class RandomProjectedTangentSpace(ProjectedTangentSpace):
  def __init__(self):
    pass

#rather than random sampling for projection, do something smarter...
class OptimizedProjectedTangentSpace(ProjectedTangentSpace):
  pass

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
  
