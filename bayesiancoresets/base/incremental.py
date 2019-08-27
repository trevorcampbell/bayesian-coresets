import numpy as np
from .iterative import IterativeCoreset
from ..util.errors import NumericalPrecisionError 

#for when the coreset is built one point at a time
class IncrementalCoreset(IterativeCoreset):

  def __init__(self, **kw):
    super().__init__(**kw)
    self._check_first_itr_monotone = True

  #step = search for next best, add it, update weights in some way
  def _step(self, sz, itr):
    #search for the next best point
    f = self._select()
    if not isinstance(f, np.integer) or f < 0:
      raise ValueError(self.alg_name+'._step(): _select() must return a nonnegative integer. type = ' + str(type(f)) + ' val = ' + str(f))

    #keep a record of previous setting in case the below update fails
    prev_error = self.error()
    prev_wts = self.wts.copy()
    prev_idcs = self.idcs.copy()

    #compute and update new weights
    self._reweight(f) 

    #check to make sure our error didn't increase
    error = self.error()
    if self._check_first_itr_monotone and error > prev_error:
      #revert
      self._overwrite(prev_idcs, prev_wts)
      raise NumericalPrecisionError('Error not monotone: curr error = ' + str(error) + ' prev error = ' + str(prev_error))

    #done

  def _select(self):
    raise NotImplementedError

  def _reweight(self, f):
    raise NotImplementedError

#for when the weight update is just (1-g)*old weight + g*new weight
class ConvexUpdateIncrementalCoreset(IncrementalCoreset):

  def _reweight(self, f):
    alpha, beta = self._step_coeffs(f)

    #compute new weights/indices from alpha,beta
    new_wts = self.wts*alpha
    new_idcs = self.idcs.copy()
    if f in self.idcs:
      idx = np.where(self.idcs == f)[0][0]
      new_wts[idx] = max(0., new_wts[idx]+beta)
    else:
      new_wts = np.append(new_wts, max(0., beta))
      new_idcs = np.append(new_idcs, f)

    #update the weights
    self._overwrite(new_idcs, new_wts)

    #done

  def _step_coeffs(self, f):
    raise NotImplementedError

