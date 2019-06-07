from scipy.optimize import nnls
import numpy as np
from ..base.iterative import GreedyCoreset
from ..util.errors import NumericalPrecisionError
from .hilbert import HilbertCoreset
from .. import TOL


class OrthoPursuitCoreset(GreedyCoreset, HilbertCoreset):
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
    self.optimize()
    return
