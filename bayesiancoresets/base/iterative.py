import numpy as np
import ..utils.warn
from .coreset import Coreset
from .errors import NumericalPrecisionError

class IterativeCoreset(Coreset):
  def __init__(self, **kw):
    super().__init__(**kw)
    self.itrs = 0
    self.has_initialized = False

  def reset(self):
    super().reset()
    self.itrs = 0
    self.has_initialized = False

  def _build(self, M):
    if not self.has_initialized:
      self.itrs = self.size() #set the itrs to the initialized size of M
      self.has_initialized = True
    #print('currently ' + str(self.size()) + ' coreset pts')
    self._set_stop_point(M)
    retried_already = False
    while not self._stop():
      try:
        self._step()
        retried_already = False #refresh retried flag after a successful step
        self.itrs += 1
      except NumericalPrecisionError: #a special error type for this library denoting possibly reaching numeric precision limit
        if retried_already:
          warn.warn(self.alg_name+'._step(): iterative step failed a second time. Assuming numeric limit reached.')
          self.reached_numeric_limit = True
          break
        else:
          warn.warn(self.alg_name+'._step(): iterative step failed. Stabilizing and retrying...')
          retried_already = True
          self._stabilize()
      if self.reached_numeric_limit:
        break
    #done

  def _stop(self):
    raise NotImplementedError()
 
  def _set_stop_point(self, M):
    raise NotImplementedError()

  def _step(self):
    raise NotImplementedError()

  def _stabilize(self):
    pass #implementation optional; try to refresh cache/etc to make _step pass

class GreedyCoreset(IterativeCoreset):

  #for greedy constructions, run itrs up to M
  def _set_stop_point(self, M):
    self.itr_end = M

  def _stop(self):
    return self.itrs >= self.itr_end

  #step = search for next best, add it, update weights in some way
  def _step(self):
    #search for the next best point
    f = self._search()
    if not isinstance(f, np.integer) or f < 0:
      raise ValueError(self.alg_name+'._step(): _search() must return a nonnegative integer. type = ' + str(type(f)) + ' val = ' + str(f))
    #update weights, adding the new point
    self._update_weights(f)
    

  def _search(self):
    raise NotImplementedError

  def _update_weights(self, f):
    raise NotImplementedError


class GreedySingleUpdateCoreset(GreedyCoreset):

  def _update_weights(self, f):
    alpha, beta = self._step_coeffs(f)
    #update the weights
    preverror = self.error()
    self.wts *= alpha
    #it's possible wts[f] becomes negative if beta approx -wts[f], so threshold
    idx = np.where(self.idcs == f)[0]
    self._set(f, max((self.wts[idx] if idx.shape[0] > 0 else 0.)+beta, 0))
    
    if self.error() > preverror:
      raise NumericalPrecisionError


  def _step_coeffs(self, f):
    raise NotImplementedError

