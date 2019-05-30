from ..base.iterative import NumericalPrecisionError, GreedySingleUpdateCoreset

class FrankWolfeCoreset(GreedySingleUpdateCoreset):

  def __init__(self, tangent_space):
    self.T = tangent_space

  def _search(self):
    return (self.T.residual(self.wts).dot(self.T[:]) / self.T.norms()).argmax()

  def _step_coeffs(self, f):
    nsum = self.T.norm_sum()
    nf = self.T.norms()[f]
    xw = self.T.sum_w(self.wts)
    xs = self.T.sum()
    xf = self.T[f]
    gammanum = (nsum/nf*xf - xw).dot(xs-xw)
    gammadenom = ((nsum/nf*xf-xw)**2).sum()
    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      raise NumericalPrecisionError
    return 1. - gammanum/gammadenom, nsum*gammanum/gammadenom
  
  def _initialize(self):
    f = self._search()
    self.wts[f] = self.T.norm_sum()
    self.xw = self.norm_sum*self.x[f, :]
    self.M = 1


