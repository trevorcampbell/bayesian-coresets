import numpy as np
from .coreset import IterativeCoresetConstruction

class LinearGreedy(IterativeCoresetConstruction):
  def __init__(self, _x):
    super(LinearGreedy, self).__init__(_x)
 
  def _xw_unscaled(self):
    return False

  def _step(self, use_cached_xw):
    #search for the next best point and step length
    f = self._search()
    alpha, beta = self._step_coeffs(f) 
    
    #if the line search is invalid, possibly reached numeric limit
    if alpha is None:
      #try recomputing xw from scratch and rerunning search
      self.xw = self.wts.dot(self.x)
      f = self._search()
      alpha, beta = self._step_coeffs(f) 

      #if it's still no good, we've reached the numeric limit
      if alpha is None: #gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:  
        self.reached_numeric_limit = True
        return False

    #update the weights
    self.wts *= alpha
    self.wts[f] += beta
    #apply the same update to xw
    if use_cached_xw:
      self.xw = alpha*self.xw + beta*self.x[f, :]
    else:
      self.xw = self.wts.dot(self.x)

    return True

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()

  

  
class FrankWolfe2(LinearGreedy):
  def __init__(self, _x):
    super(FrankWolfe2, self).__init__(_x)

  def _step_coeffs(self, f):
    gammanum = (self.x[f, :] - self.xw).dot(self.snorm*self.xs - self.xw)
    gammadenom = ((self.x[f, :] - self.xw)**2).sum()
    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      return None, None
    return 1. - gammanum/gammadenom, gammanum/gammadenom
  
  def _initialize(self):
    f = self._search()
    self.wts[f] = self.norm_sum
    self.xw = self.norm_sum*self.x[f, :]
    self.M = 1

class ForwardStagewise(LinearGreedy):
  def __init__(self, _x, step_fraction=0.05):
    self.step_fraction = step_fraction
    if self.step_fraction <= 0 or self.step_fraction >= 1:
      raise ValueError(self.alg_name+'.__init__(): step_fraction must be in (0, 1)')
    super(ForwardStagewise, self).__init__(_x)

  def _step_coeffs(self, f):
    beta = (self.x[f, :]).dot(self.snorm*self.xs - self.xw)
    if beta < 0.:
      return None, None
    return 1.0, self.step_fraction*beta

class Pursuit(LinearGreedy):

  def _step_coeffs(self, f):
    beta = (self.x[f, :]).dot(self.snorm*self.xs - self.xw)
    if beta < 0.:
      return None, None
    return 1.0, beta
