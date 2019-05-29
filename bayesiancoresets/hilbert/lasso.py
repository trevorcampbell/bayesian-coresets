from ..base.optimization import OptimizationCoreset
import numpy as np
from sklearn.linear_model import Lasso
from .vector import VectorCoreset

class LassoCoreset(OptimizationCoreset, VectorCoreset):

  def __init__(self, x, use_cached_xw=False):
    super().__init__(x=x, use_cached_xw=use_cached_xw, N=x.shape[0])
 
  def _xw_unscaled(self):
    return False
 
  def _max_reg_coeff(self):
    return (self.x.dot(self.snorm*self.xs)).max()/self.N

  #def _lasso_obj(self, w, reg_coeff):
  #  return 0.5*((w.dot(self.x)-self.snorm*self.xs)**2).sum()/self.N + reg_coeff*w.sum()
  
  def _optimize(self, w0, reg_coeff):
    lasso = Lasso(reg_coeff, positive=True, fit_intercept=False)
    lasso.fit(self.x.T, self.snorm*self.xs)
    return lasso.coef_

