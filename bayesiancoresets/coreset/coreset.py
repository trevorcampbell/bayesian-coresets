import numpy as np
import logging
from .. import util
from ..util.errors import NumericalPrecisionError

class Coreset(object):
  def __init__(self):
    self.alg_name = self.__class__.__name__ 
    self.log = logging.LoggerAdapter(logging.getLogger(), {"id" : self.alg_name})
    self.wts = np.array([])
    self.idcs = np.array([], dtype=np.int64)
    self.pts = np.array([])
    self._t = 0

  def size(self):
    return (self.wts > 0).sum()

  def get(self):
    return self.wts[self.wts > 0], self.pts[self.wts > 0, :], self.idcs[self.wts > 0]

  def error(self):
    raise NotImplementedError

  def build(self, sz, trace = None):
    #algs are only expected to grow coresets; if requested sz is smaller, just log a warning and return
    if self.size() >= sz:
      self.log.warning('requested coreset of size ' + str(sz) + '; coreset is already size ' + str(self.snnls.size()) + '. Returning...')
      return
    return self._build(sz, trace)

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self, trace = None):
    try:
      prev_cost = self.error()
      old_wts = self.wts.copy()
      old_idcs = self.idcs.copy()
      old_pts = self.pts.copy()
      self._optimize(trace)
      new_cost = self.error()
      if new_cost > prev_cost*(1.+util.TOL):
        raise NumericalPrecisionError('self.optimize() returned a solution with increasing error. Numeric limit possibly reached: preverr = ' + str(prev_cost) + ' err = ' + str(new_cost) + '.\n \
                                        If the two errors are very close, try running bc.util.tolerance(tol) with tol > current tol = ' + str(util.TOL) + ' before running')
    except NumericalPrecisionError as e:
      self.log.warning(e)
      self.wts = old_wts
      self.idcs = old_idcs
      self.pts = old_pts
      return

  def _optimize(self, trace):
    raise NotImplementedError

  def _build(self, sz, trace):
    raise NotImplementedError
