import numpy as np
import warnings

class Coreset(object):
  def __init__(self, N, auto_above_N = True, initial_wts_sz=10000, **kw):
    self.alg_name = self.__class__.__name__
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
    
  def reset(self):
    #don't bother resetting wts, just set the nwts to 0
    self.nwts = 0
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]
    self.reached_numeric_limit = False

  def size(self):
    return (self.wts > 0).sum()

  def weights(self):
    return self.idcs[self.wts > 0], self.wts[self.wts > 0]

  def _refresh_views(self):
    self.wts = self._wts[:self.nwts]
    self.idcs = self._idcs[:self.nwts]

  def _double_internal(self):
    self._wts.resize(self._wts.shape[0]*2)
    self._idcs.resize(self._idcs.shape[0]*2)

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

    #get difference, append, resize if necessary
    idiff = np.setdiff(np.arange(__idcs.shape[0]), i2)
    if self.nwts + idiff.shape[0] > self._wts.shape[0]:
      self._double_internal()
    self._wts[self.nwts:idiffshape] = __wts[etc]
    self._idcs[...]

    #create views
    self._refresh_views()
    
  def error(self):
    raise NotImplementedError()

  #attempt to build a coreset of size M
  def build(self, M):
    #if M is not greater than self.size, just return 
    if M <= self.size():
      warnings.warn(self.alg_name+'.build(): coreset size must be increasing; returning. size = '+str(self.size()) + ' M = '+str(M))
      return

    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
      return

    if self.N == 0:
      warnings.warn(self.alg_name+'.build(): there are no data, returning.')
      return

    #if we requested M >= N, just give all ones and return
    if M >= self.N and self.auto_above_N:
      warnings.warn(self.alg_name+'.build(): reached a number of points >= the dataset size. Returning full weights')
      self._wts = np.ones(self.N)
      self._idcs = np.arange(self.N)
      self.nwts = self.N
      #create views
      self._refresh_views()
      return

    #initialize optimization
    if self.size() == 0:
      self._initialize()
    
    #build the coreset with size at most M
    self._build(M)

    #if we reached numeric limit during the current build, warn immediately
    if self.reached_numeric_limit:
      warnings.warn(self.alg_name+'.build(): the numeric limit has been reached. No more points will be added. size = ' + str(self.size()) + ', error = ' +str(self.error()))
    #done

  #can run after building coreset to re-solve only the weight opt, not the combinatorial selection problem
  def optimize(self):
    raise NotImplementedError()

  #runs once on first call to .build() but after init (since it may add pt(s) to the coreset)
  def _initialize(self):
    pass #optional

  def _build(self, M):
    raise NotImplementedError()

  
