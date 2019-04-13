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

  def __init__(self, tol=1e-9, **kw):
    super().__init__(**kw)
    self.tol = tol

  def _prebuild(self):
    self.M_cache = [0, self.N]

    self.lmb_cache = {}
    self.lmb_cache[0] = [self._mrc(), self._mrc()]
    self.lmb_cache[self.N] = [0., 0.]

    self.w_cache = {} 
    self.w_cache[0] = np.zeros(self.N)
    self.w_cache[self.N] = self.all_data_wts
    
    

  def _build(self, M): 
    if M > self.N:
      M = self.N
    #if we previously cached a relevant result, return
    cache_hit = self._cache_weight_update(M)
    if cache_hit:
      return M

    #otherwise do bisection search on regularization
    
    Mr = self.M_cache[bisect.bisect_right(self.M_cache, M)]
    Ml = self.M_cache[bisect.bisect_left(self.M_cache, M)-1]

    #set max lambda = the minimum val of lambda s.t. M < desired M
    lmbu = self.lmb_cache[Ml][0]
    #set min lambda = the maximum val of lambda s.t. M > desired M
    lmbl = self.lmb_cache[Mr][1]
    w = self.w_cache[Mr] if abs(Mr - M) < abs(Ml - M) else self.w_cache[Ml]
    nnz = -1
    #keep searching if 
    # 1) we haven't found M, and
    # 2) the upper/lower reg bounds are far apart in a relative sense, and
    # 3) the upper/lower reg bounds are far apart in an abolute sense (only check if lmbl == 0.)
    try:
      if nnz != M and (lmbu-lmbl)/lmbu > self.tol and (lmbu > self.tol or lmbl > 0.):
        pass
    except:
      sys.stderr.write('nnz: ' + str(nnz) + '\n')
      sys.stderr.write('lmbu: ' + str(lmbu) + '\n')
      sys.stderr.write('lmbl: ' + str(lmbl) + '\n')
      sys.stderr.write('tol: ' + str(self.tol) + '\n')
      raise

    while nnz != M and (lmbu-lmbl)/lmbu > self.tol and (lmbu > self.tol or lmbl > 0.):
      #pick new lambda
      lmb = (lmbu+lmbl)/2.

      #optimize weights 
      w = self._optimize(w, lmb)
 
      #threshold to 0
      w[w < self.tol] = 0.
      
      #add to the cache
      nnz = (w > 0).sum()
      bisect.insort(self.M_cache, nnz)
      self.w_cache[nnz] = w
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

    return self.M

  def _cache_weight_update(self, M, lower_fallback=False):
    if M in self.M_cache:
        self.M = M
        self._update_weights(self.w_cache[M])
        #optimize weights without regularization
        self.optimize()
        return True
    elif lower_fallback:
        #find closest entry in M_cache s.t. <= M
        self.M = self.M_cache[bisect.bisect_left(self.M_cache, M)-1]
        self._update_weights(self.w_cache[self.M])
        self.optimize()
        return True
    else:
        return False

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





