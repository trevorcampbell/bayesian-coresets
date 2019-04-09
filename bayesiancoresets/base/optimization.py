import numpy as np
import warnings
from scipy.special import erfc
import bisect
from .coreset import Coreset
import sys

#class OptimizationResult(object):
#  def __init__(self, x, f0, v0, f1, v1):
#    self.x = x
#    self.f0 = f0
#    self.v0 = v0
#    self.f1 = f1
#    self.v1 = v1

class OptimizationCoreset(Coreset):

  def reset(self):
    super().reset()

  def _prebuild(self):
    self.lmb_cache = [self._mrc(), 0.]
    self.w_cache = [np.zeros(self.N), np.ones(self.N)]
    self.M_cache = [0, self.N]

  def _build(self, M): 
    #if we previously cached a relevant result, return
    if M in self.M_cache:
        cached_idx = self.M_cache.index(M)
        self._update_weights(self.w_cache[cached_idx])
        self.M = M
        return M

    #otherwise do bisection search on regularization
    idx = bisect.bisect(self.M_cache, M)
    lmbu = self.lmb_cache[idx-1]
    lmbl = self.lmb_cache[idx]
    w = self.w_cache[idx] if abs(self.M_cache[idx] - M) < abs(self.M_cache[idx-1] - M) else self.w_cache[idx-1]
    nnz = -1
    itr = 0
    while nnz != M and lmbu > 0 and (lmbu-lmbl)/lmbu > 1e-6:
      itr += 1
      #pick new lambda
      lmb = (lmbu+lmbl)/2.

      #optimize weights 
      w = self._optimize(w, lmb)
      
      #add to the cache
      nnz = (w > 0).sum()
      idx = bisect.bisect(self.M_cache, nnz)
      self.lmb_cache.insert(idx, lmb)
      self.w_cache.insert(idx, w)
      self.M_cache.insert(idx, nnz)
      if nnz < M:
        lmbu = lmb
      else:
        lmbl = lmb
 
    #find closest entry in M_cache s.t. <= M
    idx = bisect.bisect(self.M_cache, M)
    self.M = self.M_cache[idx-1]
    self._update_weights(self.w_cache[idx-1])
    return self.M

  def _mrc(self):
    if not hasattr(self, 'mrcoeff'):
      if not np.all(self.wts == 0):
        raise ValueError()
      self.mrcoeff = self._max_reg_coeff()
    return self.mrcoeff
      
  def _max_reg_coeff(self):
    raise NotImplementedError()
  
  def _optimize(self, w0, reg_coeff):
    raise NotImplementedError()


  #removed since optimize() should be specified in the objective-type parent class (e.g. kl, vector)
  #def optimize(self, check_obj_decrease=False, verbose=False):
  #  res = self._optimize(self.wts, self.wts > 0, 0.)  
  #  w = res.x
  #  f0 = res.f0
  #  v0 = res.v0
  #  f1 = res.f1
  #  v1 = res.v1 

  #  #update weights to optimized version
  #  if check_obj_decrease:
  #    diffmean = f1 - f0
  #    diffvar =  v1 + v0
  #    #check if gaussian with mean diffmean, diffvar > 0, only update if pr > 0.5
  #    pr_decrease = 0.5*erfc(diffmean / (np.sqrt(2*diffvar)))
  #    if pr_decrease > 0.5:
  #      self.wts = w
  #  else:
  #    self.wts = w


def adam(x0, grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8):
  x = x0.copy()
  adam_m1 = np.zeros(x.shape[0])
  adam_m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    adam_m1 = adam_b1*adam_m1 + (1.-adam_b1)*g
    adam_m2 = adam_b2*adam_m2 + (1.-adam_b2)*g**2
    upd = adam_a1/(i+1+adam_a2)*adam_m1/(1.-adam_b1**(i+1))/(adam_eps + np.sqrt(adam_m2/(1.-adam_b2**(i+1))))
    x -= upd

    #project onto x>=0
    x = np.maximum(x, 0.)

  return x





