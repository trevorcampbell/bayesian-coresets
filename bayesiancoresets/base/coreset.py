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

  def _resize_internal(self, sz):
    self._wts.resize(sz)
    self._idcs.resize(sz)

  def _set(self, _idcs, _wts):
    if _idcs.shape[0] != _wts.shape[0]:
      raise ValueError(self.alg_name + '._set(): new idcs and wts must have the same shape')
    if np.any(_wts < 0) or np.any(_idcs < 0) or not np.issubdtype(_idcs.dtype, np.integer):
      raise ValueError(self.alg_name+'._set(): new weights + idcs must be nonnegative, and new idcs must have integer type')
    #reuse old memory if possible
    if self._wts.shape[0] < _wts.shape[0]:
      self._wts.resize(_wts.shape[0])
      self._idcs.resize(_idcs.shape[0])
    #update internal rep
    self.nwts = _wts.shape[0]
    self._wts[:_wts.shape[0]] = _wts
    self._idcs[:_idcs.shape[0]] = _idcs
    #create views
    self._refresh_views()
    
  def _add(self, idx):
    if not isinstance(idx, np.integer) or idx < 0:
      raise ValueError(self.alg_name+'._set(): new coreset point must have nonnegative integer index')
    #expand memory if necessary
    if self.nwts == self._wts.shape[0]:
      self._resize_internal(self._wts.shape[0]*2)
    #set the new index internally
    self._idcs[self.nwts] = idx
    self._wts[self.nwts] = 0.
    self.nwts += 1
    #create new views
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

  
