import numpy as np
import warnings

class Coreset(object):
  def __init__(self, N, auto_above_N = True, **kw):
    self.alg_name = self.__class__.__name__
    self.auto_above_N = auto_above_N
    self.N = N
    self.M = 0
    self.reached_numeric_limit = False
    self.all_data_wts = np.ones(self.N)
    self.wts = np.zeros(N)
    
  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.reached_numeric_limit = False

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

    #if we requested M >= N, just give all ones and return
    if M >= self.N and self.auto_above_N:
      self._update_weights(self.all_data_wts)
      self.M = self.N
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

  def _update_weights(self, w):
    self.wts = w
    self._update_cache()

  #gets called when wts updated
  def _update_cache(self):
    pass
    
  #runs once on first call to .build()
  def _prebuild(self):
    pass #implementation optional

  def _build(self, M):
    raise NotImplementedError()

  
