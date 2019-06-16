import numpy as np
from bayesiancoresets.base import IterativeCoreset
from bayesiancoresets.base import SamplingCoreset
from bayesiancoresets.base import SingleGreedyCoreset
from bayesiancoresets.base import OptimizationCoreset

def dummy_sampler(wts, n_samples, D):
  return np.random.randn(n_samples, D)

def dummy_potentials(samples, N):
  return np.ones((N, samples.shape[0]))

class DummySampler(object):
  def __init__(self, D):
    self.D = D
  def sample(self, wts, n_samples):
    return np.random.randn(n_samples, self.D)

class DummyOptimizationCoreset(OptimizationCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _optimize(self, w, lmb):
    return np.zeros(self.N)

  def _max_reg_coeff(self):
    return 1.

  def optimize(self):
    pass

class DummySamplingCoreset(SamplingCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _compute_sampling_probabilities(self):
    return np.ones(self.N)/self.N
  
  def _update_cache(self):
    pass

class DummyIterativeCoreset(IterativeCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _step(self):
    return True

class DummySingleGreedyCoreset(SingleGreedyCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _search(self):
    return np.random.randint(self.N)

  def _step_coeffs(self, f):
    return 1., 1.



