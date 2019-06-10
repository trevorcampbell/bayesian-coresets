import numpy as np
from ..base.sampling import SamplingCoreset
from ..util.errors import NumericalPrecisionError
from .. import TOL
from .kl import KLCoreset


class ImportanceSamplingKLCoreset(KLCoreset,SamplingCoreset):
  def __init__(self, N, tangent_space_factory):
    super().__init__(N=N) 
    self.tsf = tangent_space_factory
    self.step_sched = lambda i : step_size*np.sqrt(1./(1.+i))

  def _compute_sampling_probabilities(self):
    return np.ones(self.N) #TODO something better here

class UniformSamplingKLCoreset(ImportanceSamplingKLCoreset):
  def _compute_sampling_probabilities(self):
    return np.ones(self.N)



