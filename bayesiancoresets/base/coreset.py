import numpy as np
import logging
import secrets
from .. import util

class Coreset(object):
  def __init__(self, N, auto_above_N = True, initial_wts_sz=1000, repeat_logs=False, **kw):
    #self.alg_name = '.'.join([b.__name__ for b in self.__class__.__bases__]) + '.' + self.__class__.__name__ + '-'+secrets.token_hex(3)
    self.alg_name = self.__class__.__name__ + '-'+secrets.token_hex(3)
    self.log = logging.getLogger(self.alg_name)
    self.log.setLevel(util.LOGLEVEL)
    util.add_handler(self.log, repeat_logs)
    self.auto_above_N = auto_above_N
    self.N = N
    self.reached_numeric_limit = False
    self.nwts = 0
    #internal reps of wts and idcs
    self._wts = np.zeros(initial_wts_sz)
    self._idcs = np.zeros(initial_wts_sz, dtype=np.int64)
    #outward facing views
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]

  def __del__(self):
    if self.log.handlers[0].n_suppressed > 0:
      self.log.warning('suppressed ' + str(self.log.handlers[0].n_suppressed) + ' warnings')
    
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

  #overwrite any wts at __idcs, append any new ones
  def _set(self, __idcs, __wts):

    __idcs = np.atleast_1d(__idcs)
    __wts = np.atleast_1d(__wts)
    if __idcs.shape[0] != __wts.shape[0]:
      raise ValueError(self.alg_name + '._set(): new idcs and wts must have the same shape')
    if np.any(__wts < 0) or np.any(__idcs < 0) or not np.issubdtype(__idcs.dtype, np.integer):
      raise ValueError(self.alg_name+'._set(): new weights + idcs must be nonnegative, and new idcs must have integer type')
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

  def error(self):
    raise NotImplementedError()

  #attempt to build a coreset of size M
  def build(self, M):
    #if M is not greater than self.size, just return 
    if M <= self.size():
      self.log.warning('coreset size must be increasing; returning. size = '+str(self.size()) + ' M = '+str(M))
      return

    if self.reached_numeric_limit:
      self.log.warning('the numeric limit was already reached; returning. size = ' + str(self.size()) + ', error = ' +str(self.error()))
      return

    if self.N == 0:
      self.log.warning('there are no data, returning.')
      return

    #if we requested M >= N, just give all ones and return
    if M >= self.N and self.auto_above_N:
      self.log.warning('reached a number of points >= the dataset size. Returning full weights. M = ' + str(M) + ' N = ' + str(self.N))
      self._wts = np.ones(self.N)
      self._idcs = np.arange(self.N)
      self.nwts = self.N
      #create views
      self._refresh_views()
      self.reached_numeric_limit = True
      return

    #initialize optimization
    if self.size() == 0:
      self._initialize()
      if M <= self.size():
        if M < self.size():
          self.log.warning('initialization created more than M = ' + str(M) + ' points: size = ' + str(self.size()))
        return #jump out early if initialization created at least M points 
      

    #build the coreset with size at most M
    self._build(M)

    #if we reached numeric limit during the current build, warn immediately
    if self.reached_numeric_limit:
      self.log.warning('the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
    #done

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    raise NotImplementedError()

  #runs once on first call to .build() but after init (since it may add pt(s) to the coreset)
  def _initialize(self):
    pass #optional

  def _build(self, M):
    raise NotImplementedError()

  
