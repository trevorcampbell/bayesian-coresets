import numpy as np
import warnings
from scipy.special import erfc
import bisect

class OptimizationCoreset(Coreset):

  def __init__(self, N, opt_itrs=1000, adam_a=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
    super(OptimizationCoreset, self).__init__(N)
    self.opt_itrs = opt_itrs
    self.adam_a = adam_a
    self.adam_b1 = adam_b1
    self.adam_b2 = adam_b2
    self.adam_eps = adam_eps

  def reset(self):
    super(OptimizationCoreset, self).reset()
    self.lmb_cache = [self._max_reg_coeff(), 0.]
    self.w_cache = [np.zeros(self.N), np.ones(self.N)]
    self.M_cache = [0, self.N]

  def _build(self, M):
    #do bisection search and keep cache of results
    cached_idx = (self.M_cache == M).nonzero()[0]
    if cached_idx.size > 0:
      self.wts = self.w_cache[ cached_idx[0] ]
      self.M = M
      return M

    idx = bisect(self.M_cache, M)
    lmbu = self.lmb_cache[idx]
    lmbl = self.lmb_cache[idx-1]
    wi = abs(self.M_cache[idx] - M) < abs(self.M_cache[idx-1] - M) ? self.w_cache[idx] : self.w_cache[idx-1]
    nnz = -1
    while nnz != M and (lmbu-lmbl)/lmbu > 1e-6:
      #pick new lambda
      lmb = (lmbu+lmbl)/2.

      #optimize weights
      w = self._adam(wi, lambda w : self._grad(w, np.full(self.N, True), reg(lmb))[0]) 
      
      #add to the cache
      nnz = (w > 0).sum()
      idx = bisect.bisect(self.M_cache, nnz)
      self.lmb_cache.insert(idx, lmb)
      self.w_cache.insert(idx, w)
      self.M_cache.insert(idx, nnz)

    #find closest entry in M_cache to M
    idx = bisect(self.M_cache, M)
    Mu = self.M_cache[idx]
    Ml = self.M_cache[idx-1]
    if abs(Mu - M) < abs(Ml-M):
      self.wts = self.w_cache[idx]
      self.M = Mu
      return Mu
    else:
      self.wts = self.w_cache[idx-1]
      self.M = Ml
      return Ml

  def _max_reg_coeff(self):
    raise NotImplementedError()
  
  #support stochastic: output estimate, uncertainty = variance
  #support caching
  def _reg(self, w, idcs, coeff):
    raise NotImplementedError()

  #support stochastic: output estimate, uncertainty = variance
  #support caching
  def _obj(self, w, idcs, regularization=None):
    raise NotImplementedError()

  #support stochastic: output estimate, uncertainty = variance
  #support caching
  def _grad(self, w, idcs, regularization=None):
    raise NotImplementedError()

  def weights(self):
    return self.wts

  def error(self):
    return self._obj_estimate(True)

  def _adam(self, w, grad):
    w = w.copy() #avoid overwriting input, just in case
    adam_m1 = np.zeros(w.shape[0])
    adam_m2 = np.zeros(w.shape[0])
    for i in range(opt_itrs):
      g = grad(w)
      adam_m1 = self.adam_b1*adam_m1 + (1.-self.adam_b1)*g
      adam_m2 = self.adam_b2*adam_m2 + (1.-self.adam_b2)*g**2
      upd = self.adam_a(i)*adam_m1/(1.-self.adam_b1**(i+1))/(self.adam_eps + np.sqrt(adam_m2/(1.-self.adam_b2**(i+1))))
      w -= upd

      #project onto w>=0
      w = np.maximum(wi, 0.)
    return w

  def optimize(self, check_obj_decrease=False, verbose=False):
    w = self._adam(self.wts, lambda w : self._grad(w, self.wts>0, None)[0])

    #update weights to optimized version
    if check_obj_decrease:
      old_obj = self._obj(self.wts, self.wts > 0, None) 
      new_obj = self._obj(w, w > 0, None) 
      diffmean = new_obj[0] - old_obj[0]
      diffvar =  old_obj[1] + new_obj[1]
      #check if gaussian with mean diffmean, diffvar > 0, only update if pr > 0.5
      pr_decrease = 0.5*erfc(diffmean / (np.sqrt(2*diffvar)))
      if pr_decrease > 0.5:
        self.wts = w
    else:
      self.wts = w







