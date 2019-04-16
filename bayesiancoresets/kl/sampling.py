import numpy as np
from .kl import KLCoreset
from ..base.sampling import SamplingCoreset

class SamplingKLCoreset(SamplingCoreset, KLCoreset):

  def _compute_sampling_probabilities(self):
    return (self._sample_potentials(np.zeros(self.N))).std(axis=1)

class UniformSamplingKLCoreset(SamplingKLCoreset):

  def _compute_sampling_probabilities(self):
    return np.ones(self.N)


