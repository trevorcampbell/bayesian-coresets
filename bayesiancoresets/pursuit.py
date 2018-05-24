from .coreset import GreedySingleUpdate

class Pursuit(GreedySingleUpdate):
  def _xw_unscaled(self):
    return False

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()

  def _step_coeffs(self, f):
    beta = (self.x[f, :]).dot(self.snorm*self.xs - self.xw)
    if beta < 0.:
      return None, None
    return 1.0, beta


