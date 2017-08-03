import numpy as np

class _FrankWolfe(object):
  def __init__(self):
    self.reset()

  def run(self, M):
    if M == 0:
      return

    if self.M == 0:
      self.wts = np.zeros(self.N)
      xs = self.x.sum(axis=0)
      scores = (xs*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      self.wts[f] = self.sig/self.norms[f]
      self.M = 1

    for m in range(self.M, M):
      xs = ((1. - self.wts)[:, np.newaxis]*self.x).sum(axis=0)
      scores = (xs*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      
      xw = (self.wts[:,np.newaxis]*self.x).sum(axis=0)
      gammanum = self.sig*scores[f]
      gammadenom = self.sig**2
      for nz in self.wts.nonzero()[0]:
        gammanum -= self.wts[nz]*(self.x[nz, :].dot(xs)) 
        gammadenom -= 2.*self.wts[nz]*self.sig/self.norms[f]*(self.x[f, :].dot(self.x[nz, :]))
        gammadenom += self.wts[nz]*(self.x[nz,:].dot(xw))
      gamma = gammanum/gammadenom
      if gamma < 0 or gamma > 1:
        print 'Warning: gamma not in [0, 1]: ' + str(gamma)
        gamma = (0. if gamma < 0 else gamma)
        gamma = (1. if gamma > 1 else gamma)
      self.wts *= (1.-gamma)
      self.wts[f] += gamma*self.sig/self.norms[f] 

    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = None

class _ImportanceSampling(object):
  def __init__(self):
    self.reset()

  def run(self, M):
    if self.ps is None:
      self.ps = self.norms/self.norms.sum()
      self.cts = np.zeros(self.N)
      
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.cts/self.ps/M
    
    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = None
    self.cts = None
    self.ps = None

class _Projection(object):
  def __init__(self, data, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, sample_approx_posterior, init_prm, N_SGD_itr, projection_type):
    self.N = data.shape[0]
    if init_prm is not None:
      self.dim = init_prm.shape[0]
    else:
      self.dim = sample_approx_posterior().shape[0]
    self.log_likelihood = log_likelihood
    self.log_prior = log_prior
    self.grad_log_likelihood = grad_log_likelihood
    self.grad_log_prior = grad_log_prior
    self.hess_log_joint = hess_log_joint
    self.data = data
    self.projection_type = projection_type
    self.x = np.zeros((self.N, 0))
    if sample_approx_posterior:
      self.sample_approx_posterior = sample_approx_posterior
    else:
      self.sample_approx_posterior = self.get_approx_posterior(init_prm, N_SGD_itr)

    self.update_projection_dimension(projection_dim)
    return

  def get_approx_posterior(self, init_prm, n_itr):
    #run SGD to get laplace posterior approx
    prm = init_prm.copy()
    for i in range(n_itr):
      grd = self.grad_log_likelihood(self.data[np.random.randint(self.data.shape[0]), :], prm) + 1.0/self.N * self.grad_log_prior(prm)
      prm += np.sqrt(1.0/(1.0+i))*grd
    hess = self.hess_log_joint(self.data, prm)
    return lambda : np.random.multivariate_normal(prm, -np.linalg.inv(hess))

  def update_projection_dimension(self, projection_dim):
    if projection_dim < self.x.shape[1]:
      self.x = self.x[:, :projection_dim]

    if projection_dim > self.x.shape[1]:
      old_dim = self.x.shape[1]
      w = np.zeros((self.N, projection_dim))
      w[:, :old_dim] = self.x
      w *= np.sqrt(old_dim)
      if self.projection_type == 'F':
        for j in range(projection_dim-old_dim):
          w[:, j+old_dim] = np.sqrt(self.dim)*self.sample_projection_component()
      else:
        for j in range(projection_dim-old_dim):
          w[:, j+old_dim] = self.sample_projection_component()
      w /= np.sqrt(projection_dim)
      self.x = w
   
    self.norms = np.sqrt((self.x**2).sum(axis=1))
    self.sig = self.norms.sum()
    return

  def reset_projection(self, projection_dim=None):
    if not projection_dim:
      projection_dim = self.x.shape[1]

    self.update_projection_dimension(0)
    self.update_projection_dimension(projection_dim)
    return

  def sample_projection_component(self):
    if self.projection_type == 'F':
      return self.grad_log_likelihood(self.data, self.sample_approx_posterior())[:, np.random.randint(self.dim)]
    else:
      return self.log_likelihood(self.data, self.sample_approx_posterior())

class ProjectedFrankWolfe(_Projection, _FrankWolfe):
  def __init__(self, data, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, sample_approx_posterior = None, init_prm = None, N_SGD_itr = 0, projection_type = 'F'):
    _FrankWolfe.__init__(self)
    _Projection.__init__(self, data, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, sample_approx_posterior, init_prm, N_SGD_itr, projection_type)

class ProjectedImportanceSampling(_Projection, _ImportanceSampling):
  def __init__(self, data, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, sample_approx_posterior = None, init_prm = None, N_SGD_itr = 0, projection_type = 'F'):
    _ImportanceSampling.__init__(self)
    _Projection.__init__(self, data, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, sample_approx_posterior, init_prm, N_SGD_itr, projection_type)

class FullDataset(object):
  def __init__(self, N):
    self.wts = np.ones(N)

  def run(self, M):
    return

  def reset(self):
    return

  def reset_projection(self):
    return


class RandomSubsample(object):
  def __init__(self, N):
    self.ps = 1.0/float(N) * np.ones(N)
    self.N = N
    self.reset()

  def run(self, M):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.cts/self.ps/M
    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.cts = np.zeros(self.N)
  
  def reset_projection(self):
    return



