import numpy as np
from ..base.iterative import GreedySingleUpdateCoreset
from ..util.errors import NumericalPrecisionError
from .hilbert import HilbertCoreset



class MatchingPursuitCoreset(HilbertCoreset,GreedySingleUpdateCoreset):
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

  def _step_coeffs(self, f):
    alpha = 1.0
    xf = self.T[f]
    nf = self.T.norms()[f]
    beta = self.T.residual(self.wts, self.idcs).dot(xf/nf)/nf
    fidx = np.where(self.idcs == f)[0]
    if fidx.shape[0] > 1:
      raise ValueError('self.idcs == f in multiple locations: np.where(idcs==f) = ' + str(fidx) + ' f = ' + str(f))
    if fidx.shape[0] > 0 and beta < -self.wts[fidx]:
      beta = -self.wts[f]
    return alpha, beta

