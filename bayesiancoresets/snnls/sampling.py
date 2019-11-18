import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS


class ImportanceSampling(SparseNNLS):

  def __init__(self, A, b):
    super().__init__(A, b)
    self.cts = np.zeros(self.x.shape[0]) 
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps / self.ps.sum()

  def reset(self):
    super().reset()
    self.cts = np.zeros(self.x.shape[0]) 

  def _compute_sampling_probabilities(self):
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps / self.ps.sum()

  def _select(self):
    return np.random.choice(self.ps.shape[0], p = self.ps)

  def _reweight(self, f):
    self.cts[f] += 1
    self.x = (self.cts / self.cts.sum()) / self.ps 

class UniformSampling(ImportanceSampling):
  def __init__(self, A, b):
    super().__init__(A, b)
    self.ps = np.ones(self.x.shape[0]) / float(self.x.shape[0])
