import numpy as np
from .coreset import CoresetConstruction

class FullDataset(object):
  def __init__(self, x):
    self.wts = np.ones(x.shape[0])

  def run(self, M):
    return

  def reset(self):
    return

  def weights(self):
    return self.wts

  def error(self):
    return 0.


class FullDataset2(CoresetConstruction):
  def _xw_unscaled(self):
    return False

  def _initialize(self):
    self.wts = np.ones(self.N)

  def _build(self, M, use_cached_xw):
    pass

  

