from .coreset import GreedySingleUpdate

class Pursuit(GreedySingleUpdate):

  def _xw_unscaled(self):
    return False

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()

  def _step_coeffs(self, f):
    v1 = self.xw - self.xw.dot(self.x[f, :])*self.x[f, :]
    v2 = (self.xw**2).sum()*self.x[f, :] - self.xw.dot(self.x[f, :])*self.xw
 
    if (v1**2).sum() == 0.:
      return None, None
    
    alpha = (self.snorm*self.xs).dot(v1) / self.xw.dot(v1)
    beta = (self.snorm*self.xs).dot(v2) / self.xw.dot(v1)

    if beta < 0. or alpha < 0.:
      return None, None
    return alpha,  beta

  def _initialize(self):
    f = self._search()
    self.wts[f] = self.snorm*self.x[f, :].dot(self.xs)
    self.xw = self.wts[f]*self.x[f, :]
    self.M = 1

