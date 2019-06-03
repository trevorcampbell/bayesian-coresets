from ..base.optimization import OptimizationCoreset
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from .. import TOL
import warnings

#TODO just run once and use ordered list of decreasing magnitude

#run lasso on normalized vectors
class LassoCoreset(OptimizationCoreset):
  def __init__(self, tangent_space):
    super().__init__(N=tangent_space.num_vectors()) 
    self.log.warning('LASSO + bisection regularization search implementation is not yet stable. Be wary of results!')
    self.T = tangent_space
    if np.any(self.T.norms() == 0):
      raise ValueError(self.alg_name+'.__init__(): tangent space must not have any 0 vectors')
 
  #TODO debug why this isn't correct
  #for now just 1000* as a bandaid
  def _max_reg_coeff(self):
    return 1000*((self.T[:]/self.T.norms()[:,np.newaxis]).dot(self.T.sum())).max()/self.N

  #def _lasso_obj(self, w, reg_coeff):
  #  return 0.5*((w.dot(self.x)-self.snorm*self.xs)**2).sum()/self.N + reg_coeff*w.sum()
  
  def _optimize(self, w0, idx, reg_coeff):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      lasso = Lasso(reg_coeff, positive=True, fit_intercept=False, tol=TOL)
      lasso.fit((self.T[:]/self.T.norms()[:,np.newaxis]).T, self.T.sum())
      return lasso.coef_[lasso.coef_ > TOL]/self.T.norms()[lasso.coef_ > TOL], np.where(lasso.coef_ > TOL)[0]

  def optimize(self):
    #run least squares optimal weight update
    X = self.T[self.idcs]
    res = nnls(X.T, self.T.sum())
 
    #if the optimizer failed or our cost increased, stop
    prev_cost = self.error()
    if res[1] >= prev_cost:
      raise NumericalPrecisionError('nnls returned a solution with increasing error. Numeric limit reached: preverr = ' + str(prev_cost) + ' err = ' + str(res[1]))

    #update weights, xw, and prev_cost
    self._overwrite(self.idcs.copy(), res[0])
    return

  def error(self):
    return self.T.error(self.wts, self.idcs)


