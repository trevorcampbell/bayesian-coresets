import numpy as np
from .coreset import Coreset


#TODO FIX for new riemann structure
#also make useful in the large N setting (cts cannot all be stored)
class SamplingCoreset(Coreset):

  def __init__(self, sampling_probabilities=None, **kw):
    super().__init__(**kw)
    if sampling_probabilities is None:
      self.ps = self._compute_sampling_probabilities()
    else:
      self.ps = sampling_probabilities
    if np.any(self.ps < 0.):
      raise ValueError(self.alg_name+'.__init__(): sampling probabilities must be all nonnegative')
    self.ps /= self.ps.sum()
    self.cts = np.zeros(self.N)
    
  def reset(self):
    super().reset()
    self.cts = np.zeros(self.N)
    
  def _build(self, M):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self._update_weights(self._weight_scaling()*self.cts/self.ps/M)
    return M

  def _compute_sampling_probabilities(self):
    raise NotImplementedError()
  
