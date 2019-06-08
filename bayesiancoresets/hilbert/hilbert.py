import numpy as np
from scipy.optimize import nnls
from ..util.errors import NumericalPrecisionError
from .. import TOL

class HilbertCoreset(object):
  
  def optimize(self):
    prev_cost = self.error()
    old_wts = self.wts.copy()
    old_idcs = self.idcs.copy()

    #run least squares optimal weight update
    X = self.T[self.idcs]
    res = nnls(X.T, self.T.sum())

    #if the optimizer failed or our cost increased, stop
    if res[1] >= prev_cost:
      self._overwrite(old_idcs, old_wts)
      raise NumericalPrecisionError('nnls returned a solution with increasing error. Numeric limit reached: preverr = ' + str(prev_cost) + ' err = ' + str(res[1]))

    #update weights, xw, and prev_cost
    self._overwrite(self.idcs.copy(), res[0])
    return
    
  def error(self):
    return self.T.error(self.wts, self.idcs)

  

