import numpy as np
import logging
import secrets
from .. import util
from ..util.errors import NumericalPrecisionError

class Coreset(object):
  def __init__(self):
    self.alg_name = self.__class__.__name__ + '-'+secrets.token_hex(3)
    self.log = logging.LoggerAdapter(logging.getLogger(), {"id" : self.alg_name})
    self.reached_numeric_limit = False
    self.wts = np.array([])
    self.idcs = np.array([], dtype=np.int64)
    self.pts = np.array([])

  def reset(self):
    self.wts = np.array([])
    self.idcs = np.array([], dtype=np.int64)
    self.pts = np.array([])
    self.reached_numeric_limit = False

  def size(self):
    return (self.wts > 0).sum()

  def get(self):
    if self.wts.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    return self.wts[self.wts > 0], self.pts[self.wts > 0, :], self.idcs[self.wts > 0]

  def error(self):
    raise NotImplementedError()

  def build(self, itrs):
    if self.reached_numeric_limit:
      return

    if itrs <= 0:
      return

    self._build(itrs)

    #if we reached numeric limit during the current build, warn
    if self.reached_numeric_limit:
      self.log.warning('the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    try:
      prev_cost = self.error()
      old_wts = self.wts.copy()
      old_idcs = self.idcs.copy()
      old_pts = self.pts.copy()
      self._optimize()
      new_cost = self.error()
      if new_cost > prev_cost*(1.+util.TOL):
        raise NumericalPrecisionError('self.optimize() returned a solution with increasing error. Numeric limit possibly reached: preverr = ' + str(prev_cost) + ' err = ' + str(new_cost) + '.\n \
                                        If the two errors are very close, try running bc.util.tolerance(tol) with tol > current tol = ' + str(util.TOL) + ' before running')
    except NumericalPrecisionError as e:
      self.log.warning(e)
      self.wts = old_wts
      self.idcs = old_idcs
      self.pts = old_pts
      self.reached_numeric_limit = True
      return

  def _optimize(self):
    raise NotImplementedError

  def _build(self, itrs):
    raise NotImplementedError
