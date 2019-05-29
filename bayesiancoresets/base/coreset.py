import numpy as np
import warnings

class Coreset(object):
  def __init__(self, N, auto_above_N = True, **kw):
    self.alg_name = self.__class__.__name__
    self.auto_above_N = auto_above_N
    self.N = N
    self.M = 0
    self.reached_numeric_limit = False
    self.wts = []
    self.idcs = []
    
  def reset(self):
    self.M = 0
    self.wts = []
    self.idcs = []
    self.reached_numeric_limit = False

  def size(self):
    return len(self.wts)

  def weights(self):
    return np.array(self.idcs, dtype=np.int64), np.array(self.wts)

  def error(self):
    raise NotImplementedError()

  
  #Tangent space:
  #GIGA: M = # iterations
  #L1: M = coreset size

  #Riemann Exact (i.e. constant tangent space updates):
  #Greedy Exact grads: M = # iters
  #Greedy Stochastic grads: same
  #L1 stochastic grads: M = coreset size

  #Riemann Infrequent Tangent Updates
  #Sequential Hilbert (in each round, run hilbert coreset in a tangent space. Move to new space from output w. Do some decayed step): M = coreset size. Specify # iterations specially in the object. Default to running to convergence.
  #Greedy Hilbert + Quadratic: (in each round, pick vec greedily, minimize quadratic approx of KL): M = #iters 

  #attempt to build a coreset of size M
  #guaranteed to output a coreset of size <= M
  def build(self, M):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      warnings.warn(self.alg_name+'.build(): M must be increasing; returning. self.M = '+str(self.M) + ' M = '+str(M))
      return self.M

    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return self.M

    if self.N == 0:
      warnings.warn(self.alg_name+'.build(): there are no data, returning.')
      return self.M

    #if we requested M >= N, just give all ones and return
    if M >= self.N and self.auto_above_N:
      warnings.warn(self.alg_name+'.build(): reached a number of points >= the dataset size. Returning full weights')
      self.wts = [1]*self.N
      self.idcs = list(range(self.N))
      self.M = self.N
      return self.M

    #initialize optimization
    if self.M == 0:
      Mnew = self._initialize()
      if Mnew != int(Mnew):
        raise ValueError(self.alg_name + '.build(): ._initialize(M) must return the initial number of steps taken. type = ' + str(type(Mnew)) + ' val = ' + str(Mnew))
      self.M = Mnew
    
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
  def _initialize(self):
    return 0 #special implementation optional; default do nothing and keep coreset size at 0

  def _build(self, M):
    raise NotImplementedError()

  
