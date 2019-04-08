from .pursuit import MatchingPursuitCoreset

class ForwardStagewiseCoreset(MatchingPursuitCoreset):
  def __init__(self, _x, use_cached_xw=False, step_fraction=0.05):
    super().__init__(_x, use_cached_xw)
    self.step_fraction = step_fraction
    if self.step_fraction <= 0 or self.step_fraction >= 1:
      raise ValueError(self.alg_name+'.__init__(): step_fraction must be in (0, 1)')

  def _step_coeffs(self, f):
    alpha = 1.0
    beta = (self.snorm*self.xs - self.xw).dot(self.x[f,:])
    if beta < -self.wts[f]:
      beta = -self.wts[f]
    return alpha, self.step_fraction*beta


#old code before using MP as a parent
#from .coreset import GreedySingleUpdate
#
#class ForwardStagewise(GreedySingleUpdate):
#  def __init__(self, _x, step_fraction=0.05):
#    self.step_fraction = step_fraction
#    if self.step_fraction <= 0 or self.step_fraction >= 1:
#      raise ValueError(self.alg_name+'.__init__(): step_fraction must be in (0, 1)')
#    super(ForwardStagewise, self).__init__(_x)
#
#  def _xw_unscaled(self):
#    return False
#
#  def _search(self):
#    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()
#
#  def _step_coeffs(self, f):
#    beta = (self.x[f, :]).dot(self.snorm*self.xs - self.xw)
#    if beta < 0.:
#      return None, None
#    return 1.0, self.step_fraction*beta


