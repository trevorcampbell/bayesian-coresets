import numpy as np
from .geometry import *
import warnings
from .frankwolfe import FrankWolfe

class ForwardStagewise(FrankWolfe):
  def __init__(self, _x, step_fraction):
    self.step = step_fraction
    if self.step <= 0 or self.step >= 1:
      raise ValueError('ForwardStagewise.__init__(): step_fraction must be in (0, 1)')
    super(ForwardStagewise, self).__init__(_x)

  #options are fast, accurate (fast tracks xw and wts separately, accurate updates xw from wts at each iter)
  def run(self, M, update_method='fast'):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('ForwardStagewise.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or self.reached_numeric_limit:
      warnings.warn('ForwardStagewise.run(): either data has no nonzero vectors or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    for m in range(self.M, M):
      #search for best data point and compute line search
      f = self.search()
      gamma = self.step_size(f)
      if gamma < 0:
        break

      #reduce step size for stagewise mode
      gamma *= self.step

      self.wts *= (1.-gamma)
      self.wts[f] += gamma*self.sig/self.norms[f] 
      if update_method == 'fast':
        self.xw = (1.-gamma)*self.xw + gamma*self.sig/self.norms[f]*self.x[f, :]
        self.f_update += 1.
      else:
        self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)
        self.f_update += (self.wts > 0).sum()
      self.M = m+1

    return

  def exp_bound(self, M=None):
    raise NotImplementedError()

  def sqrt_bound(self, M=None):
    raise NotImplementedError()

