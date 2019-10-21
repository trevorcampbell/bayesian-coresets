import numpy as np
from scipy.optimize import nnls
from ..util.errors import NumericalPrecisionError

class HilbertCoreset(object):

  def _optimize(self):
    X = self.T[self.idcs]
    res = nnls(X.T, self.T.sum(), maxiter=100*X.shape[0])
    self._overwrite(self.idcs.copy(), res[0])
    return

  def error(self):
    return self.T.error(self.wts, self.idcs)


