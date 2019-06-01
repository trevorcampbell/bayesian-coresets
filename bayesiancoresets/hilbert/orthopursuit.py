from scipy.optimize import nnls
import numpy as np
from ..base.iterative import GreedyCoreset
from ..util.errors import NumericalPrecisionError
from .. import TOL



class OrthoPursuitCoreset(GreedyCoreset):
  def __init__(self, tangent_space):
    super().__init__(N=tangent_space.num_vectors()) 
    self.T = tangent_space
    if np.any(self.T.norms() == 0):
      raise ValueError(self.alg_name+'.__init__(): tangent space must not have any 0 vectors')

  def _search(self):
    dots = (self.T[:]/self.T.norms()[:,np.newaxis]).dot(self.T.residual(self.wts, self.idcs))

    #if no active indices, just output argmax
    if self.idcs.shape[0] == 0:
      return dots.argmax()
    
    #search positive direction on whole dataset, negative direction on active set
    fpos = dots.argmax()
    pos = dots[fpos]
    fneg = (-dots[self.idcs]).argmax()
    neg = (-dots[self.idcs])[fneg]

    if pos >= neg:
      return fpos
    else:
      return self.idcs[fneg]

  def _update_weights(self, f):

    old_wts = self.wts.copy()

    f_already = np.where(self.idcs == f)[0].shape[0] > 0

    #check to make sure value to add is not in the current set (error should be ortho to current subspace)
    if f_already:
      raise NumericalPrecisionError('search selected a nonzero weight to update.')
    else:
      self._set(f, 1.)
      
    #run least squares optimal weight update
    X = self.T[self.idcs]
    res = nnls(X.T, self.T.sum())
 
    #if the optimizer failed or our cost increased, stop
    prev_cost = self.error()
    if res[1] >= prev_cost:
      self.wts = old_wts
      raise NumericalPrecisionError('nnls returned a solution with increasing error. Numeric limit reached: preverr = ' + str(prev_cost) + ' err = ' + str(res[1]))

    #update weights, xw, and prev_cost
    self._set(self.idcs, res[0])
    
    return
    
  def error(self):
    return self.T.error(self.wts, self.idcs)

  

