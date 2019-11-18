import numpy as np
import logging
import secrets
from .. import util
from ..util.errors import NumericalPrecisionError
from scipy.optimize import nnls

class SparseNNLS(object):
  def __init__(self, A, b, check_error_monotone = True):
    self.alg_name = self.__class__.__name__ + '-'+secrets.token_hex(3)
    self.log = logging.LoggerAdapter(logging.getLogger(), {"id" : self.alg_name})
    self.A = A
    self.b = b
    self.reached_numeric_limit = False
    self.w = np.zeros(A.shape[1])
    self.check_error_monotone = check_error_monotone

  def reset(self):
    self.w = np.zeros(self.A.shape[1])
    self.reached_numeric_limit = False

  def size(self):
    return (self.w > 0).sum()

  def weights(self):
    return self.w.copy()

  def error(self):
    return np.sqrt(((self.A.dot(self.w) - self.b)**2).sum())

  def build(self, itrs):
    if self.reached_numeric_limit:
      self.log.warning('the numeric limit was already reached; returning. size = ' + str(self.size()) + ', error = ' +str(self.error()))
      return

    if self.A.size == 0:
      self.log.warning('there are no data, returning.')
      return

    retried_already = False
    for i in range(itrs):
      try:
        #keep a record of previous setting in case the below update fails
        size_nonzero = self.size() > 0 #create a flag here, since ._reweight(f) will change this
        if self.check_error_monotone and size_nonzero:
          prev_error = self.error()
          prev_w = self.w.copy()
    
        #search for the next best point
        f = self._select()

        #compute and update new weights
        self._reweight(f) 

        #check to make sure our error didn't increase (only if we have run at least 1 itr, since most algs have a setup step on the 1st itr)
        if self.check_error_monotone and size_nonzero:
          error = self.error()
          if error > prev_error: 
            #revert
            self.w = prev_w
            raise NumericalPrecisionError('Error not monotone: curr error = ' + str(error) + ' prev error = ' + str(prev_error))
          retried_already = False #refresh retried flag after a successful step
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

    #if we reached numeric limit during the current build, warn
    if self.reached_numeric_limit:
      self.log.warning('the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
    #done

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    try:
      prev_cost = self.error()
      prev_w = self.w.copy()
      nz_idcs = self.w > 0
      res = nnls(self.A[:, nz_idcs], self.b, maxiter=100*self.A.shape[1])
      self.w[nz_idcs] = res[0]
      new_cost = self.error()
      if new_cost > prev_cost*(1.+util.TOL):
        raise NumericalPrecisionError('self.optimize() returned a solution with increasing error. Numeric limit possibly reached: preverr = ' + str(prev_cost) + ' err = ' + str(new_cost) + '.\n \
                                        If the two errors are very close, try running bc.util.tolerance(tol) with tol > current tol = ' + str(util.TOL) + ' before running')
    except NumericalPrecisionError as e:
      self.log.warning(e)
      self.w = prev_w
      self.reached_numeric_limit = True
      return

  def _stabilize(self):
    pass #implementation optional; try to refresh cache/etc to make _step pass

  def _select(self):
    raise NotImplementedError

  def _reweight(self, f):
    raise NotImplementedError
