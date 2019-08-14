import numpy as np
from .coreset import Coreset
from ..util.errors import NumericalPrecisionError 
from .. import util
import bisect
import sys

class OptimizationCoreset(Coreset):

  def __init__(self, adam_a1 = 1., adam_a2 = 1., **kw):
    super().__init__(**kw)
    self.log.warning('OptimizationCoreset implementation is not yet stable. Be wary of results!')
    self.adam_a1 = adam_a1
    self.adam_a2 = adam_a2

  def _initialize(self, M):
    self.M_cache = [0, self.N]

    self.lmb_cache = {}
    self.lmb_cache[0] = [self._mrc(), self._mrc()]
    self.lmb_cache[self.N] = [0., 0.]

    self.w_cache = {} 
    self.w_cache[0] = np.array([])
    self.w_cache[self.N] = np.ones(self.N)

    self.idx_cache = {} 
    self.idx_cache[0] = np.array([], dtype=np.int64)
    self.idx_cache[self.N] = np.arange(self.N)

  def _build(self, M): 
    if M > self.N:
      M = self.N
    #if we previously cached a relevant result, return
    cache_hit = self._cache_weight_update(M)
    if cache_hit:
      return

    #otherwise do bisection search on regularization
    
    Mr = self.M_cache[bisect.bisect_right(self.M_cache, M)]
    Ml = self.M_cache[bisect.bisect_left(self.M_cache, M)-1]

    #set max lambda = the minimum val of lambda s.t. M < desired M
    lmbu = self.lmb_cache[Ml][0]
    #set min lambda = the maximum val of lambda s.t. M > desired M
    lmbl = self.lmb_cache[Mr][1]
    w = self.w_cache[Mr] if abs(Mr - M) < abs(Ml - M) else self.w_cache[Ml]
    idx = self.idx_cache[Mr] if abs(Mr - M) < abs(Ml - M) else self.idx_cache[Ml]
    nnz = -1
    #keep searching if 
    # 1) we haven't found M, and
    # 2) the upper/lower reg bounds are far apart in a relative sense, and
    # 3) the upper/lower reg bounds are far apart in an abolute sense (only check if lmbl == 0.)

    while nnz != M and (lmbu-lmbl)/lmbu > util.TOL and (lmbu > util.TOL or lmbl > 0.):
      #pick new lambda
      lmb = (lmbu+lmbl)/2.

      #optimize weights 
      w, idx = self._optimize(w, idx, lmb)
 
      #threshold to 0
      idx = idx[w > util.TOL]
      w = w[w>util.TOL]
      
      
      #add to the cache
      nnz = idx.shape[0]
      bisect.insort(self.M_cache, nnz)
      self.w_cache[nnz] = w
      self.idx_cache[nnz] = idx
      if nnz in self.lmb_cache:
        self.lmb_cache[nnz][0] = min(lmb, self.lmb_cache[nnz][0])
        self.lmb_cache[nnz][1] = max(lmb, self.lmb_cache[nnz][1])
      else:
        self.lmb_cache[nnz] = [lmb, lmb]

      if nnz < M:
        lmbu = lmb
      else:
        lmbl = lmb
 
    self._cache_weight_update(M, lower_fallback=True)
    return

  def _cache_weight_update(self, M, lower_fallback=False):
    if M in self.M_cache:
        self._overwrite(self.idx_cache[M], self.w_cache[M])
        #optimize weights without regularization
        self.optimize()
        return True
    elif lower_fallback:
        #find closest entry in M_cache s.t. <= M
        M = self.M_cache[bisect.bisect_left(self.M_cache, M)-1]
        self._overwrite(self.idx_cache[M], self.w_cache[M])
        self.optimize()
        return True
    else:
        return False

  def _mrc(self):
    if not hasattr(self, 'mrcoeff'):
      if not self.idcs.size == 0:
        raise ValueError('Tried to initialize max_reg_coeff with nonempty weights (after already building)')
      self.mrcoeff = self._max_reg_coeff()
    return self.mrcoeff
      
  def _max_reg_coeff(self):
    raise NotImplementedError()
  
  def _optimize(self, w0, idcs, reg_coeff):
    raise NotImplementedError()


