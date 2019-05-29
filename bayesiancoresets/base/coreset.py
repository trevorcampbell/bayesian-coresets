import numpy as np
import warnings

class Coreset(object):
  def __init__(self, N, auto_above_N = True, **kw):
    self.alg_name = self.__class__.__name__
    self.auto_above_N = auto_above_N
    self.N = N
    self.reached_numeric_limit = False
    self.wts = []
    self.idcs = []
    
  def reset(self):
    self.wts = []
    self.idcs = []
    self.reached_numeric_limit = False

  def size(self):
    return len(self.wts)

  def weights(self):
    return np.array(self.idcs, dtype=np.int64), np.array(self.wts)

  def error(self):
    raise NotImplementedError()

  #attempt to build a coreset of size M
  def build(self, M):
    #if M is not greater than self.size, just return 
    if M <= self.size():
      warnings.warn(self.alg_name+'.build(): coreset size must be increasing; returning. size = '+str(self.size()) + ' M = '+str(M))
      return

    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
      return

    if self.N == 0:
      warnings.warn(self.alg_name+'.build(): there are no data, returning.')
      return

    #if we requested M >= N, just give all ones and return
    if M >= self.N and self.auto_above_N:
      warnings.warn(self.alg_name+'.build(): reached a number of points >= the dataset size. Returning full weights')
      self.wts = [1]*self.N
      self.idcs = list(range(self.N))
      return

    #initialize optimization
    if self.size() == 0:
      self._initialize()
    
    #build the coreset with size at most M
    self._build(M)

    #if we reached numeric limit during the current build, warn immediately
    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
    #done

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    raise NotImplementedError()

  #runs once on first call to .build() but after init (since it may add pt(s) to the coreset)
  def _initialize(self):
    pass #optional

  def _build(self, M):
    raise NotImplementedError()

  
