import numpy as np
from scipy.optimize import lsq_linear, minimize
import warnings
from .vector import VectorCoreset
from ..base.iterative import IterativeCoreset

class OrthoPursuitCoreset(VectorCoreset, IterativeCoreset):

  def __init__(self, x, use_cached_xw=False):
    super().__init__(x=x, use_cached_xw=use_cached_xw, N=x.shape[0])

  def _xw_unscaled(self):
    return False

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()
  
  def _step(self):
    #search for FW vertex and compute line search
    f = self._search()

    #check to make sure value to add is not in the current set (error should be ortho to current subspace)
    if self.wts[f] > 0:
      warnings.warn(self.alg_name+'.run(): search selected a nonzero weight to update')

    #run least squares optimal weight update
    active_idcs = self.wts > 0
    active_idcs[f] = True
    X = self.x[active_idcs, :]
    res = lsq_linear(X.T, self.snorm*self.xs, bounds=(0., np.inf), max_iter=max(1000, 10*self.xs.shape[0]))
 
    #if the optimizer failed or our cost increased, stop
    prev_cost = self.error()
    if not res.success or np.sqrt(2.*res.cost) >= prev_cost:
      self.reached_numeric_limit = True
      return False

    #update weights, xw, and prev_cost
    self.wts[active_idcs] = res.x
    self.xw = self.wts.dot(self.x)
    
    return True

  

