import numpy as np
from .coreset import Coreset

class FullDataset(Coreset):

  def _initialize(self):
    self.wts = np.ones(self.N)

  def weights(self):
    return self.wts

  def error(self):
    return 0.

  def _build(self, M):
    return M

  def optimize(self):
    pass


