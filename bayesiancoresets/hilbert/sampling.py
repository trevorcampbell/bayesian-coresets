import numpy as np
from ..base.sampling import SamplingCoreset
from ..util.errors import NumericalPrecisionError
from .. import TOL
from .hilbert import HilbertCoreset


class ImportanceSamplingHilbertCoreset(HilbertCoreset,SamplingCoreset):

  def __init__(self, tangent_space):
    self.T = tangent_space
    if np.any(self.T.norms() == 0):
      raise ValueError(self.alg_name+'.__init__(): tangent space must not have any 0 vectors')
    super().__init__(N=tangent_space.num_vectors()) 

  def _compute_sampling_probabilities(self):
    if self.T.norms_sum() > 0.:
      return self.T.norms()
    else:
      return np.ones(self.N)

class UniformSamplingHilbertCoreset(ImportanceSamplingCoreset):
  def _compute_sampling_probabilities(self):
    return np.ones(self.N)



