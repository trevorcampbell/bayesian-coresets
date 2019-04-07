import numpy as np
from .vector import VectorCoreset
from ..base.sampling import SamplingCoreset

class VectorSamplingCoreset(VectorCoreset, SamplingCoreset):

  def _xw_unscaled(self):
    return False

  def _compute_sampling_probabilities(self):
    if self.norm_sum > 0.:
      self.ps = self.norms/self.norm_sum
    else:
      self.ps = 1.0/float(self.N) * np.ones(self.N)

  def _update_cache(self):
    self.wts *= self.norms #puts the weights on the scale of the normalized vectors
    self.xw = self.wts.dot(self.x) #computes new cached xw

class VectorUniformSamplingCoreset(VectorSamplingCoreset)

  def _compute_sampling_probabilities(self):
    self.ps = 1.0/float(self.N) * np.ones(self.N)


