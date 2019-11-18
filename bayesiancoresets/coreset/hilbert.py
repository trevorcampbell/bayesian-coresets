import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA

class HilbertCoreset(Coreset):
  def __init__(self, loglike, N, J, snnls = GIGA):
    vecs = loglike(np.arange(N), J)
    vecs -= vecs.mean(axis=1)[:, np.newaxis]
    self.snnls = snnls(vecs.T, vecs.sum(axis=0))

  def _build(self, itrs, sz = None):
    self.snnls.build(itrs)
    self._overwrite(*self.snnls.weights())

  def _optimize(self):
    self.snnls.optimize()

  def error(self):
    return self.snnls.error()
