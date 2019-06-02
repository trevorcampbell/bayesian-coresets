import numpy as np
from .coreset import Coreset

class FullDataCoreset(Coreset):

  def _initialize(self):
    self._overwrite(np.arange(self.N), np.ones(self.N))

  def error(self):
    if self.size() < self.N:
      raise NotImplementedError(self.alg_name+'.error(): Error on full data coreset = 0 after build, but undefined beforehand.')
    return 0.

  def _build(self, M):
    pass

  def optimize(self):
    pass


