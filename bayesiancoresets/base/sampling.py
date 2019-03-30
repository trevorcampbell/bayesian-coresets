import numpy as np
from .coreset import Coreset

class ImportanceSamplingCoreset(Coreset):
  def __init__(self, ps):
    if np.any(ps < 0.):
      raise ValueError(self.alg_name+'.__init__(): ps must be all nonnegative')
    self.ps /= self.ps.sum()

  def _initialize(self):
    self.cts = np.zeros(self.N)

  def _build(self, M):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.norms*self.cts/self.ps/M
    return M

class UniformSamplingCoreset(ImportanceSamplingCoreset):
  def __init__(self, N):
    self.N = N
    self.ps = 1./float(N)*np.ones(N)
