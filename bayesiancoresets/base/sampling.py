import numpy as np
from .coreset import Coreset

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
    self.cts += np.random.multinomial(M - self.cts.sum(), self.ps)
    active = np.where(self.cts > 0)[0]
    self._set(active, self.cts[active]/self.ps[active]/M)

  #defaults to uniform sampling
  def _compute_sampling_probabilities(self):
    raise NotImplementedError


