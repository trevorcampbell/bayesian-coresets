import numpy as np
from ..util.errors import NumericalPrecisionError
from .snnls import SparseNNLS

class FrankWolfe(SparseNNLS):

  def __init__(self, A, b):
    super().__init__(A, b)

    self.Anorms = np.sqrt((self.A**2).sum(axis=0))
    if np.any( self.Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')
    self.An = self.A / self.Anorms

  def _select(self):
    residual = self.b - self.A.dot(self.w)
    return (self.An.T.dot(residual)).argmax()

  def _step_coeffs(self, f):
    #special case if this is the first point to add (places iterate on constraint polytope)
    if self.size() == 0:
      return 0., self.Anorms.sum() / self.Anorms[f]

    nsum = self.Anorms.sum()
    nf = self.Anorms[f]
    xw = self.A.dot(w)
    xf = self.A[:, f]

    gammanum = (nsum/nf*xf - xw).dot(self.b-xw)
    gammadenom = ((nsum/nf*xf-xw)**2).sum()

    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      raise NumericalPrecisionError('precision loss in gammanum/gammadenom: num = ' + str(gammanum) + ' denom = ' + str(gammadenom))
    return 1. - gammanum/gammadenom, nsum/nf*gammanum/gammadenom
  

