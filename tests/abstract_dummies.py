import numpy as np
from bayesiancoresets.base import IterativeCoreset
from bayesiancoresets.base import SamplingCoreset
from bayesiancoresets.base import SingleGreedyCoreset
from bayesiancoresets.base import OptimizationCoreset

class DummySampler(object):
  def __init__(self, D):
    self.D = D
  def sample(self, wts, n_samples):
    return np.random.randn(self.D, n_samples)

class DummyOptimizationCoreset(OptimizationCoreset):

  def __init__(self, N):
    super().__init__(N=N)

  def _optimize(self, w, lmb):
    return np.zeros(self.N)

  def _max_reg_coeff(self):
    return 1.

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



