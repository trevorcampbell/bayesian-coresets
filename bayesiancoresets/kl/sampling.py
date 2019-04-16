import numpy as np
from .kl import KLCoreset
from ..base.sampling import SamplingCoreset

class SamplingKLCoreset(SamplingCoreset, KLCoreset):

  #TODO: better sampling probabilities in forward/reverse cases
  def _compute_sampling_probabilities(self):
    return np.ones(N)

class UniformSamplingKLCoreset(SamplingKLCoreset):

  def _compute_sampling_probabilities(self):
    return np.ones(self.N)


