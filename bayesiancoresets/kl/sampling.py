import numpy as np
from .kl import KLCoreset
from ..base.sampling import SamplingCoreset

class KLSamplingCoreset(KLCoreset, SamplingCoreset):

  def _compute_sampling_probabilities(self):
    if np.any(self.scales > 0.):
      self.ps = self.scales/self.scales.sum()
    else:
      self.ps = 1.0/float(self.N) * np.ones(self.N)

  def _update_cache(self):
    self.wts *= self.scales #puts the weights on the same scale as the scaled potentials

class KLUniformSamplingCoreset(KLSamplingCoreset):

  def _compute_sampling_probabilities(self):
    self.ps = 1.0/float(self.N) * np.ones(self.N)


