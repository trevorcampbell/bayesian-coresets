import numpy as np
from .coreset import IterativeCoresetConstruction

class LinearGreedy(IterativeCoresetConstruction):
  def __init__(self, _x):
    super(LinearGreedy, self).__init__(_x)
 
  def _xw_unscaled(self):
    return False

  def _step(self, use_cached_xw):
    #search for the next best point and step length
    f, gamma = self._get_step()
    #if the step size was bad, set numeric limit reached and quit
    if gamma < 0:
      self.reached_numeric_limit = True
      return False

    #shrink the step
    gamma *= self.step_fraction

    #update the weights
    self.wts *= (1.-gamma)
    self.wts[f] += gamma*self.weight_scale
    #apply the same update to xw
    if use_cached_xw:
      self.xw = (1.-gamma)*self.xw + gamma*self.weight_scale*self.x[f, :]
    else:
      self.xw = self.wts.dot(self.x)

    return True

  
  def _get_step(self):
    f = self._search()
    gammanum, gammadenom = self._step_coeffs(f) 
    
    #if the line search is invalid, possibly reached numeric limit
    #try recomputing xw from scratch and rerunning search
    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      self.xw = self.wts.dot(self.x)
      f = self._search()
      gammanum, gammadenom = self._step_coeffs(f) 

      #if it's still no good, we've reached the numeric limit
      if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:  
        return f, -1

    return f, gammanum/gammadenom

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()

  def _step_coeffs(self, f):
    gammanum = (self.weight_scale*self.x[f, :] - self.xw).dot(self.snorm*self.xs - self.xw)
    gammadenom = ((self.weight_scale*self.x[f, :] - self.xw)**2).sum()
    return gammanum, gammadenom

  
class FrankWolfe2(LinearGreedy):
  def __init__(self, _x):
    super(FrankWolfe2, self).__init__(_x)
    self.weight_scale = self.norm_sum
    self.step_fraction = 1.0
  
  def _initialize(self):
    f = self._search()
    self.wts[f] = self.norm_sum
    self.xw = self.norm_sum*self.x[f, :]
    self.M = 1

class ForwardStagewise(LinearGreedy):
  def __init__(self, _x, step_fraction=0.05):
    self.weight_scale = 1.0
    self.step_fraction = step_fraction
    if self.step_fraction <= 0 or self.step_fraction >= 1:
      raise ValueError(self.alg_name+'.__init__(): step_fraction must be in (0, 1)')
    super(ForwardStagewise2, self).__init__(_x)

  def _initialize(self):
    pass

class Pursuit(LinearGreedy):
  def __init__(self, _x):
    super(Pursuit, self).__init__(_x)
    self.step_fraction = 1.0
    self.weight_scale = 1.0

  def _initialize(self):
    pass
