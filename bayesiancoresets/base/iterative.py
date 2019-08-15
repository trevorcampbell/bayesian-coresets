import numpy as np
from .coreset import Coreset
from ..util.errors import NumericalPrecisionError 

class IterativeCoreset(Coreset):

  def __init__(self, **kw):
    super().__init__(**kw)
    self._itr = 0
  
  def reset(self):
    super().reset()
    self._itr = 0

  def _build(self, sz, itrs):
    itr_limit = self._itr + itrs
    retried_already = False
    while self._itr < itr_limit and (not self._terminate_on_size() or self.size() < sz):
      try:
        self._step(sz, self._itr)
        retried_already = False #refresh retried flag after a successful step
        self._itr += 1
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

  def _terminate_on_size(self):
    return True
 
  def _step(self, sz, itr):
    raise NotImplementedError()

  def _stabilize(self):
    pass #implementation optional; try to refresh cache/etc to make _step pass

