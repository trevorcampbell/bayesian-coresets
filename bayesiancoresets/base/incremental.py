import numpy as np
from .coreset import Coreset
from ..util.errors import NumericalPrecisionError 

class IncrementalCoreset(Coreset):

  def _build(self, sz, itrs):
    
    retried_already = False
    while not self._stop():
      try:
        self._step()
        retried_already = False #refresh retried flag after a successful step
        self.itrs += 1
      except NumericalPrecisionError as e: #a special error type for this library denoting possibly reaching numeric precision limit
        self.log.warning('numerical precision error: ' + str(e))
        if retried_already:
          self.log.warning('iterative step failed a second time. Assuming numeric limit reached.')
          self.reached_numeric_limit = True
          break
        else:
          self.log.warning('iterative step failed. Stabilizing and retrying...')
          retried_already = True
          self._stabilize()
      if self.reached_numeric_limit:
        break
    #done

  def _stop(self):
    raise NotImplementedError()
 
  def _set_desired_coreset_size(self, M):
    raise NotImplementedError()

  def _step(self):
    raise NotImplementedError()

  def _stabilize(self):
    pass #implementation optional; try to refresh cache/etc to make _step pass

class GreedyCoreset(IterativeCoreset):

  #for greedy constructions, run itrs up to M
  def _set_desired_coreset_size(self, M):
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
    #keep a record of previous setting in case the below update fails
    preverror = self.error()
    prevwts = self.wts.copy()
    previdcs = self.idcs.copy()
    prevnwts = self.nwts
    #update the weights
    self.wts *= alpha
    #it's possible wts[f] becomes negative if beta approx -wts[f], so threshold
    idx = np.where(self.idcs == f)[0]
    self._update(f, max((self.wts[idx] if idx.shape[0] > 0 else 0.)+beta, 0))
    error = self.error()
    if error > preverror:
      #revert
      self._overwrite(previdcs, prevwts)
      #self.wts = prevwts
      #self.idcs = previdcs
      #self.nwts = prevnwts
      raise NumericalPrecisionError('Error not monotone: curr error = ' + str(error) + ' prev error = ' + str(preverror))

  def _step_coeffs(self, f):
    raise NotImplementedError

