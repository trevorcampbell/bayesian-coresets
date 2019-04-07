import numpy as np
from .coreset import Coreset

class SamplingCoreset(Coreset):

  def _initialize(self):
    self.ps = self._compute_sampling_probabilities()
    if np.any(self.ps < 0.):
      raise ValueError(self.alg_name+'.__init__(): sampling probabilities must be all nonnegative')
    self.ps /= self.ps.sum()
    
  def reset(self):
    self.cts = np.zeros(self.N)
    super(SamplingCoreset, self).reset()
    
  def _build(self, M):
    self.cts += np.random.multinomial(M - self.M, self.ps)
    self.wts = self.cts/self.ps/M
    self._update_cache()
    return M

  def weights(self):
    return self.wts

  def _compute_sampling_probabilities(self):
    raise NotImplementedError()
  
  def _update_cache(self):
    pass


