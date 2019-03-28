import numpy as np
from .coreset import Coreset

#TODO this is not fully implemented yet
class ImportanceSamplingCoreset(Coreset):
  def __init__(self, N):
    raise NotImplementedError

  def _initialize(self):
    self.cts = np.zeros(self.N)
    self.ps = self.norms/self.norm_sum

  def _build(self, M):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.norms*self.cts/self.ps/M
    return M

class UniformSamplingCoreset(ImportanceSamplingCoreset):
  def _initialize(self):
    self.cts = np.zeros(self.N)
    self.ps = 1.0/float(self.N)*np.ones(self.N)


