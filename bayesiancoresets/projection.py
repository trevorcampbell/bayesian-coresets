import numpy as np

class _Projection(object):
  def __init__(self, projection_dim, sample_approx_posterior):
    self.N = self.data.shape[0]
    self.dim = sample_approx_posterior().shape[0]
    self.x = np.zeros((self.N, 0))
    self.sample_approx_posterior = sample_approx_posterior
    self.update_projection_dimension(projection_dim)
    return

  def update_projection_dimension(self, projection_dim):
    if projection_dim < self.x.shape[1]:
      self.x = self.x[:, :projection_dim]

    if projection_dim > self.x.shape[1]:
      old_dim = self.x.shape[1]
      w = np.zeros((self.N, projection_dim))
      w[:, :old_dim] = self.x
      w *= np.sqrt(old_dim)
      for j in range(projection_dim-old_dim):
          w[:, j+old_dim] = self.sample_projection_component()
      w /= np.sqrt(projection_dim)
      self.x = w
    return

  def reset_projection(self, projection_dim=None):
    if not projection_dim:
      projection_dim = self.x.shape[1]

    self.update_projection_dimension(0)
    self.update_projection_dimension(projection_dim)
    return

  def get(self):
    return self.x.copy()
  
class Projection2(_Projection):
  def __init__(self, data, log_likelihood, projection_dim, sample_approx_posterior):
    self.data = data
    self.log_likelihood = log_likelihood
    _Projection.__init__(self, projection_dim, sample_approx_posterior)

  def sample_projection_component(self):
    return self.log_likelihood(self.data, self.sample_approx_posterior())

class ProjectionF(_Projection):
  def __init__(self, data, grad_log_likelihood, projection_dim, sample_approx_posterior):
    self.data = data
    self.grad_log_likelihood = grad_log_likelihood
    _Projection.__init__(self, projection_dim, sample_approx_posterior)
  
  def sample_projection_component(self):
    return np.sqrt(self.dim)*self.grad_log_likelihood(self.data, self.sample_approx_posterior(), np.random.randint(self.dim))

