import numpy as np
from .coreset import Coreset

class FullDataCoreset(Coreset):

  def _prebuild(self):
    self.wts = np.ones(self.N)
    self.M = self.N

  def weights(self):
    return self.wts

  def error(self):
    if self.M < self.N:
      raise NotImplementedError(self.alg_name+'.error(): Error on full data coreset = 0 after build, but undefined beforehand.')
    return 0.

  def _build(self, M):
    return self.N


  def optimize(self):
    pass


