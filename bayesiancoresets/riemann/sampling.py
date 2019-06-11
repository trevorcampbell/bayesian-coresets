import numpy as np
from ..base.sampling import SamplingCoreset
from ..util.errors import NumericalPrecisionError
from .kl import KLCoreset


class ImportanceSamplingKLCoreset(KLCoreset,SamplingCoreset):
  def __init__(self, N, tangent_space_factory, step_sched = lambda i : np.sqrt(1./(1.+i))):
    super().__init__(N=N) 
    self.tsf = tangent_space_factory
    self.step_sched = step_sched

  def _compute_sampling_probabilities(self):
    return np.ones(self.N) #TODO something better here

class UniformSamplingKLCoreset(ImportanceSamplingKLCoreset):
  def _compute_sampling_probabilities(self):
    return np.ones(self.N)



