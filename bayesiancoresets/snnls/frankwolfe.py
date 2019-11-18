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

  def _reweight(self, f):
    if self.size() == 0:
      #special case if this is the first point to add (places iterate on constraint polytope)
      alpha = 0.
      beta = self.Anorms.sum() / self.Anorms[f]
    else:
      nsum = self.Anorms.sum()
      nf = self.Anorms[f]
      xw = self.A.dot(self.w)
      xf = self.A[:, f]

      gammanum = (nsum/nf*xf - xw).dot(self.b-xw)
      gammadenom = ((nsum/nf*xf-xw)**2).sum()

      if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
        raise NumericalPrecisionError('precision loss in gammanum/gammadenom: num = ' + str(gammanum) + ' denom = ' + str(gammadenom))

      alpha = 1. - gammanum/gammadenom
      beta = nsum/nf*gammanum/gammadenom

    self.w = alpha*self.w
    self.w[f] = max(0., self.w[f]+beta)

  

