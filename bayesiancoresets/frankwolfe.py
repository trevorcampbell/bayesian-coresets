from .vector import SingleGreedyVectorCoreset
from .iterative import NumericalPrecisionError

class FrankWolfeCoreset(SingleGreedyVectorCoreset):

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()

  def _xw_unscaled(self):
    return False

  def _step_coeffs(self, f):
    gammanum = (self.norm_sum*self.x[f, :] - self.xw).dot(self.snorm*self.xs - self.xw)
    gammadenom = ((self.norm_sum*self.x[f, :] - self.xw)**2).sum()
    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      raise NumericalPrecisionError
    return 1. - gammanum/gammadenom, self.norm_sum*gammanum/gammadenom
  
  def _initialize(self):
    f = self._search()
    self.wts[f] = self.norm_sum
    self.xw = self.norm_sum*self.x[f, :]
    self.M = 1


