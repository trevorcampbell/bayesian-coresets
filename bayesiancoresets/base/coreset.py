import numpy as np
import warnings

class Coreset(object):
  def __init__(self, N, **kw):
    self.alg_name = self.__class__.__name__
    self.N = N
    self.M = 0
    self.reached_numeric_limit = False
    self.wts = np.zeros(self.N)
    
  def reset(self):
    self.M = 0
    self.reached_numeric_limit = False
    self.wts = np.zeros(self.N)

  def size(self):
    return (self.wts > 0).sum()

  def weights(self):
    raise NotImplementedError()

  def error(self):
    raise NotImplementedError()

  def build(self, M):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      warnings.warn(self.alg_name+'.build(): M must be increasing; returning. self.M = '+str(self.M) + ' M = '+str(M))
      return self.M
    if self.reached_numeric_limit:
      return self.M

    if self.N == 0:
      warnings.warn(self.alg_name+'.build(): there are no data, returning.')
      return self.M

    #initialize optimization
    if self.M == 0:
      self._prebuild()
    
    #build the coreset with size at most M
    Mnew = self._build(M)
    if Mnew != int(Mnew):
      raise ValueError(self.alg_name + '.build(): ._build(M) must return the new total number of steps taken. type = ' + str(type(Mnew)) + ' val = ' + str(Mnew))
    self.M = Mnew

    #if we reached numeric limit during the current build, warn immediately
    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. M = ' + str(self.M) + ', error = ' +str(self.error()))
    return self.M

  def optimize(self):
    raise NotImplementedError()
    
  #runs once on first call to .build()
  def _prebuild(self):
    pass #implementation optional

  def _build(self, M):
    raise NotImplementedError()


