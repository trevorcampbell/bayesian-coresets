import numpy as np
from .coreset import CoresetConstruction

class FullDataset(CoresetConstruction):
  def _xw_unscaled(self):
    return False

  def _initialize(self):
    self.wts = self.norms.copy()
    self.xw = self.wts.dot(self.x)

  def _build(self, M, use_cached_xw):
    pass

  

