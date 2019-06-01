from ..base.optimization import OptimizationCoreset
import numpy as np
from sklearn.linear_model import Lasso
from .vector import VectorCoreset
from .. import TOL


#run lasso on normalized vectors
class LassoCoreset(OptimizationCoreset):
  def __init__(self, tangent_space):
    super().__init__(N=tangent_space.num_vectors()) 
    self.T = tangent_space
    if np.any(self.T.norms() == 0):
      raise ValueError(self.alg_name+'.__init__(): tangent space must not have any 0 vectors')
 
  def _max_reg_coeff(self):
    return ((self.T[:]/self.T.norms()[:,np.newaxis]).dot(self.T.sum())).max()/self.N

  #def _lasso_obj(self, w, reg_coeff):
  #  return 0.5*((w.dot(self.x)-self.snorm*self.xs)**2).sum()/self.N + reg_coeff*w.sum()
  
  def _optimize(self, w0, idx, reg_coeff):
    lasso = Lasso(reg_coeff, positive=True, fit_intercept=False)
    lasso.fit((self.T[:]/self.T.norms()[:,np.newaxis]).T, self.T.sum())
    return lasso.coef_[lasso.coef_ > TOL], np.where(lasso.coef_ > TOL)[0]

  def error(self):
    return self.T.error(self.wts, self.idcs)


