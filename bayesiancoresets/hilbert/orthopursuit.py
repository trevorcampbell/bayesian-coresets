from scipy.optimize import nnls
import numpy as np
from ..base.iterative import GreedyCoreset
from ..util.errors import NumericalPrecisionError
from .hilbert import HilbertCoreset
from .. import TOL


class OrthoPursuitCoreset(HilbertCoreset,GreedyCoreset):
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

    #store prev weights/idcs before adding f
    old_wts = self.wts.copy()
    old_idcs = self.idcs.copy()

    #check to make sure value to add is not in the current set (error should be ortho to current subspace)
    #otherwise add a 0 entry to enable nnls below to use f
    f_already = np.where(self.idcs == f)[0].shape[0] > 0
    if f_already:
      raise NumericalPrecisionError('search selected a nonzero weight to update.')
    else:
      self._update(f, 0.)
   
    #run nnls, catch a numerical precision error, reset to old wts/idcs if needed, reraise to tell outer algorithms we failed
    try:
      self.optimize()
    except NumericalPrecisionError as e:
      self._overwrite(old_idcs, old_wts)
      raise

    return
