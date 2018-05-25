import numpy as np
from .coreset import CoresetConstruction

class ImportanceSampling(CoresetConstruction):

  def _xw_unscaled(self):
    return False

  def _initialize(self):
    self.cts = np.zeros(self.N)
    if self.norm_sum > 0.:
      self.ps = self.norms/self.norm_sum
    else:
      self.ps = 1.0/float(self.N) * np.ones(self.N)

  def _build(self, M, use_cached_xw):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.norms*self.cts/self.ps/M
    self.xw = self.wts.dot(self.x)
    return M

class RandomSubsampling(ImportanceSampling):
  def _initialize(self):
    self.cts = np.zeros(self.N)
    self.ps = 1.0/float(self.N)*np.ones(self.N)
