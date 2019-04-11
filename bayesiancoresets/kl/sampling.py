import numpy as np
from .kl import KLCoreset
from ..base.sampling import SamplingCoreset

class KLSamplingCoreset(SamplingCoreset, KLCoreset):

  def _compute_sampling_probabilities(self):
    if np.any(self.scales > 0.):
      return self.scales.copy()
    else:
      return np.ones(self.N)

  def _weight_scaling(self):
    return self.scales #puts the weights on the same scale as the scaled potentials

class KLUniformSamplingCoreset(KLSamplingCoreset):

  def _compute_sampling_probabilities(self):
    return np.ones(self.N)


