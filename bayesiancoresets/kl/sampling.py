import numpy as np
from .kl import KLCoreset
from ..base.sampling import SamplingCoreset

class KLSamplingCoreset(SamplingCoreset, KLCoreset):

  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True, normalized = True):
    super().__init__(potentials=potentials, sampler=sampler, n_samples=n_samples, reverse=reverse, n_lognorm_disc=n_lognorm_disc, scaled=scaled, normalized=normalized, N=len(potentials))

  def _compute_sampling_probabilities(self):
    if np.any(self.scales > 0.):
      self.ps = self.scales
    else:
      self.ps = np.ones(self.N)

  def _update_cache(self):
    self.wts *= self.scales #puts the weights on the same scale as the scaled potentials

class KLUniformSamplingCoreset(KLSamplingCoreset):

  def _compute_sampling_probabilities(self):
    self.ps = np.ones(self.N)


