from ..base.optimization import OptimizationCoreset
import numpy as np
from sklearn.linear_model import Lasso

class LassoCoreset(OptimizationCoreset, VectorCoreset):
 
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

