import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class HilbertCoreset(Coreset):
  def __init__(self, tangent_space_factory, snnls = GIGA, **kw):
    vecs = tangent_space_factory()
    self.snnls = snnls(vecs.T, vecs.sum(axis=0))
    super().__init__(**kw)

  def reset(self):
    self.snnls.reset()
    super().reset()

  def _build(self, itrs, sz):
    if self.snnls.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.snnls.size()) + ' desired sz: ' + str(sz))
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self._overwrite(w[w>0], np.where(w>0)[0])

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self._overwrite(w[w>0], np.where(w>0)[0])

  def error(self):
    return self.snnls.error()
