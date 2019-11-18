import numpy as np
import logging
import secrets
from .. import util
from ..util.errors import NumericalPrecisionError

class Coreset(object):
  def __init__(self, initial_wts_sz=1000):
    self.alg_name = self.__class__.__name__ + '-'+secrets.token_hex(3)
    self.log = logging.LoggerAdapter(logging.getLogger(), {"id" : self.alg_name})
    self.reached_numeric_limit = False
    self.nwts = 0
    #internal reps of wts and idcs
    self._wts = np.zeros(initial_wts_sz)
    self._idcs = np.zeros(initial_wts_sz, dtype=np.int64)
    #outward facing views
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]

  def reset(self):
    #don't bother resetting wts, just set the nwts to 0
    self.nwts = 0
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]
    self.reached_numeric_limit = False

  def size(self):
    return (self.wts > 0).sum()

  def weights(self):
    return self.wts[self.wts > 0], self.idcs[self.wts > 0]

  def _refresh_views(self):
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]

  def _double_internal(self):
    self.wts = None
    self.idcs = None
    self._wts.resize(self._wts.shape[0]*2)
    self._idcs.resize(self._idcs.shape[0]*2)
    self._refresh_views()

  #overwrite any wts at __idcs (keeping old values if unmodified), append any new ones
  def _update(self, __wts, __idcs):
    __idcs = np.atleast_1d(__idcs)
    __wts = np.atleast_1d(__wts)
    if __idcs.shape[0] != __wts.shape[0]:
      raise ValueError(self.alg_name + '._set(): new idcs and wts must have the same shape. idcs.shape = ' + str(__idcs.shape[0]) + ' wts.shape = ' + str(__wts.shape[0]))
    if np.any(__wts < 0) or np.any(__idcs < 0) or not np.issubdtype(__idcs.dtype, np.integer):
      raise ValueError(self.alg_name+'._set(): new weights + idcs must be nonnegative, and new idcs must have integer type. any(wts < 0) = ' + str(np.any(__wts < 0)) + ' any(idcs < 0) = ' + str(np.any(__idcs<0)) + ' dtype = ' + str(__idcs.dtype) + ' idcs = ' + str(__idcs) + ' wts = ' + str(__wts))
    #get intersection, overwrite
    inter, i1, i2 = np.intersect1d(self.idcs, __idcs, return_indices=True)

    self.wts[i1] = __wts[i2]

    #get difference, append, resizing if necessary
    idiff = np.setdiff1d(np.arange(__idcs.shape[0]), i2)
    while self.nwts + idiff.shape[0] > self._wts.shape[0]:
      self._double_internal()
    self._idcs[self.nwts:self.nwts+idiff.shape[0]] = __idcs[idiff]
    self._wts[self.nwts:self.nwts+idiff.shape[0]] = __wts[idiff]
    self.nwts += idiff.shape[0]

    #create views
    self._refresh_views()

  #completely overwrite; forget any previous weight settings
  def _overwrite(self, __wts, __idcs):
    __idcs = np.atleast_1d(__idcs)
    __wts = np.atleast_1d(__wts)
    if __idcs.shape[0] != __wts.shape[0]:
      raise ValueError(self.alg_name + '._set(): new idcs and wts must have the same shape. idcs.shape = ' + str(__idcs.shape[0]) + ' wts.shape = ' + str(__wts.shape[0]))
    if np.any(__wts < 0) or np.any(__idcs < 0) or not np.issubdtype(__idcs.dtype, np.integer):
      raise ValueError(self.alg_name+'._set(): new weights + idcs must be nonnegative, and new idcs must have integer type. any(wts < 0) = ' + str(np.any(__wts < 0)) + ' any(idcs < 0) = ' + str(np.any(__idcs<0)) + ' dtype = ' + str(__idcs.dtype) + ' idcs = ' + str(__idcs))
    #full overwrite
    while __wts.shape[0] > self._wts.shape[0]:
      self._double_internal()
    self._wts[:__wts.shape[0]] = __wts
    self._idcs[:__idcs.shape[0]] = __idcs
    self.nwts = __wts.shape[0]
    self._refresh_views()

  def error(self):
    raise NotImplementedError()

  #build of desired size sz using at most itrs iterations
  #always returns a coreset of size <= sz
  def build(self, itrs, sz):

    if self.reached_numeric_limit:
      return

    if sz < self.size():
      raise ValueError(self.alg_name+'.build(): requested coreset of size < the current size, but cannot shrink coresets; returning. Requested size = ' + str(sz) + ' current size = ' + str(self.size()))

    self._build(itrs, sz)

    #if we reached numeric limit during the current build, warn
    if self.reached_numeric_limit:
      self.log.warning('the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    try:
      prev_cost = self.error()
      old_wts = self.wts.copy()
      old_idcs = self.idcs.copy()
      self._optimize()
      new_cost = self.error()
      if new_cost > prev_cost*(1.+util.TOL):
        raise NumericalPrecisionError('self.optimize() returned a solution with increasing error. Numeric limit possibly reached: preverr = ' + str(prev_cost) + ' err = ' + str(new_cost) + '.\n \
                                        If the two errors are very close, try running bc.util.tolerance(tol) with tol > current tol = ' + str(util.TOL) + ' before running')
    except NumericalPrecisionError as e:
      self.log.warning(e)
      self._overwrite(old_wts, old_idcs)
      self.reached_numeric_limit = True
      return

  def _optimize(self):
    raise NotImplementedError

  def _build(self, itrs, sz):
    raise NotImplementedError
