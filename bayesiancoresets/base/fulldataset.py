import numpy as np
from .coreset import Coreset

class FullDataCoreset(Coreset):

  def error(self):
    if self.size() < self.N:
      raise NotImplementedError(self.alg_name+'.error(): Error on full data coreset = 0 after build, but undefined beforehand.')
    return 0.

  def _build(self, sz, itrs):
    if sz < self.N:
      self.log.warning("FullDataCoreset can't build a coreset of size < N. Returning empty coreset. Requested size = " + str(sz) + ' N = ' + str(self.N))
      return
    self._overwrite(np.arange(self.N), np.ones(self.N))

  #overwrite the base coreset class optimize to do nothing (since either empty or full / no error)
  def optimize(self):
    pass


