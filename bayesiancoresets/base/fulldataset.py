import numpy as np
from .coreset import Coreset

class FullDataCoreset(Coreset):

  #for the full dataset, weights are always just all 1s
  #need to override reset() to avoid the base coreset from setting wts = M = 0
  def reset(self):
    super(FullDataCoreset, self).reset()
    self.wts = np.ones(self.N)
    self.M = self.N

  def weights(self):
    return self.wts

  def error(self):
    return 0.

  def _build(self, M):
    return M

  def optimize(self):
    pass


