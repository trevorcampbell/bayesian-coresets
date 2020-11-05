from scipy.optimize import nnls
import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS


class OrthoPursuit(SparseNNLS):

  def __init__(self, A, b):
      """
      Initialize b

      Args:
          self: (todo): write your description
          A: (int): write your description
          b: (int): write your description
      """
    super().__init__(A, b)

    Anorms = np.sqrt((self.A**2).sum(axis=0))
    if np.any( Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')
    self.An = self.A / Anorms

  def _select(self):
      """
      Selects the next vector

      Args:
          self: (todo): write your description
      """
    residual = self.b - self.A.dot(self.w)
    dots = self.An.T.dot(residual)

    #if no active indices, just output argmax
    if self.size() == 0:
      return dots.argmax()
    
    #search positive direction on whole dataset, negative direction on active set
    fpos = dots.argmax()
    pos = dots[fpos]
    nz_idcs = self.w > 0
    fneg = (-dots[nz_idcs]).argmax()
    neg = (-dots[nz_idcs])[fneg]

    if pos >= neg:
      return fpos
    else:
      return np.arange(self.w.shape[0])[nz_idcs][fneg]

  def _reweight(self, f):
      """
      R compute the weight matrix

      Args:
          self: (todo): write your description
          f: (array): write your description
      """
    self.w[f] = 1.
    nz_idcs = self.w > 0
    res = nnls(self.A[:, nz_idcs], self.b, maxiter=100*self.A.shape[1])
    self.w[nz_idcs] = res[0]
    return




