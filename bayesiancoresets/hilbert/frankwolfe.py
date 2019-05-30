from ..base.iterative import GreedySingleUpdateCoreset
from ..base.errors import NumericalPrecisionError

class FrankWolfeCoreset(GreedySingleUpdateCoreset):

  def __init__(self, tangent_space):
    self.T = tangent_space

  def _search(self):
    return (self.T.residual(self.wts, self.idcs).dot(self.T[:]) / self.T.norms()).argmax()

  def _step_coeffs(self, f):
    nsum = self.T.norm_sum()
    nf = self.T.norms()[f]
    xw = self.T.sum_w(self.wts, self.idcs)
    xs = self.T.sum()
    xf = self.T[f]
    gammanum = (nsum/nf*xf - xw).dot(xs-xw)
    gammadenom = ((nsum/nf*xf-xw)**2).sum()
    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      raise NumericalPrecisionError
    return 1. - gammanum/gammadenom, nsum*gammanum/gammadenom
  
  def _initialize(self):
    f = self._search()
    self._add(f, self.T.norm_sum())


