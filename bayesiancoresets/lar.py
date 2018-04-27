import numpy as np
from .geometry import *
import warnings

class Lasso(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('FrankWolfe: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.norm_sqs = nrms[self.nzidcs]**2
    self.xs = self.x.sum(axis=0)
    self.N = self.x.shape[0]
    self.f_preproc = x.shape[0] + 2*self.x.shape[0] 
    self.f_search = 0.
    self.n_search = 0.
    self.f_update = 0.
    self.reached_numeric_limit = False
    self.lmb = (self.x.dot(self.xs)).max() #max lambda necessary; guarantees M = 0
    self.cached_lambdas = np.zeros((2, 2))
    self.cached_lambdas[0, 0] = self.lmb
    self.cached_lambdas[0, 1] = np.inf
    self.cached_Ms = np.zeros(2, dtype=np.int64)
    self.cached_Ms[1] = np.inf
    self.reset()

  def run(self, M, tol=1e-9, caching=True, posthoc_correct=True):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('FrankWolfe.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or self.reached_numeric_limit:
      warnings.warn('FrankWolfe.run(): either data has no nonzero vectors or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    # run bisection search to find lambda value that yields ||w||_0 = M
    # ||w||_0 is a decreasing function in lambda, with ||w||_0 = 0 when lambda >= lambda_max
    
    if caching:
      #first, get cached lambda bounds / wts for M
      Midx = np.searchsorted(self.cached_Ms, M, side='right')
      lmb_l = self.cached_lambdas[Midx, 1]
      lmb_r = self.cached_lambdas[Midx-1, 0]
      self.M = self.cached_Ms[Midx-1]
    else:
      lmb_r = self.lmb
      lmr_l = 0.

    #bisection search
    w = self.wts.copy()
    while self.M != M:
      self.lmb = 0.5*(lmb_l + lmb_r)
      
      #LASSO coordinate descent
      #initialize to previous self.M's weights
      prev_obj = 0.
      r = self.xs - self.xw 
      w[:] = self.wts
      cur_obj = (r**2).sum() + self.lmb*np.sum(np.fabs(w))
      while np.fabs((cur_obj-prev_obj)/cur_obj) > tol:
        prev_obj = cur_obj
        for j in range(w.shape[0]):
          r += self.x[j, :]*w[j]
          w[j] = max(0., (r.dot(self.x[j, :]) - self.lmb) / (self.x[j, :]**2).sum())
          r -= self.x[j, :]*w[j]
        cur_obj = (r**2).sum() + self.lmb*np.sum(np.fabs(w))
      self.M = (w > 0).sum()
      
      if self.M >= M:
        lmb_l = self.lmb
        #cache the result if we might need it in the future (only if self.M >= M)
        if caching:
          Midx = np.searchsorted(self.cached_Ms, self.M)
          if self.cached_Ms[Midx] == self.M:
            self.cached_lambdas[Midx, 0] = min(lmb, self.cached_lambdas[Midx, 0])
            self.cached_lambdas[Midx, 1] = max(lmb, self.cached_lambdas[Midx, 1])
          else:
            np.insert(self.cached_Ms, Midx, self.M)
            np.insert(self.cached_lambdas, Midx, np.ones(2)*self.lmb)
      else:
        lmb_r = self.lmb

    #w contains weights with support = M
      
    #get rid of all cached values for Ms less than self.M (since M has to always increase)
    if caching:
      keep_idcs = self.cached_Ms >= self.M
      self.cached_Ms = self.cached_Ms[keep_idcs]
      self.cached_lambdas = self.cached_lambdas[keep_idcs, :] 

    #run l-bfgs-b to get optimal weights in support
    if posthoc_correct:
      X = self.x[w > 0, :]
      q0 = w[w > 0]
      res = minimize(fun = lambda q : ((self.xs - q.dot(X))**2).sum(), 
                 x0 = q0, method='L-BFGS-B', 
                 jac = lambda q : (q.dot(X)).dot(X.T) - 2*self.xs.dot(X.T), 
                 bounds = [(0., None)]*q0.shape[0],
                 options ={'ftol': 1e-12, 'gtol': 1e-9})
      if not res.success:
        self.reached_numeric_limit = True
      else:
        w[w>0] = res.x

    #set xw and wts
    self.wts[:] = w
    self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)
    self.f_update += (self.wts > 0).sum()  
    return

  def get_num_ops(self):
    return self.f_preproc+self.f_search + self.f_update

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.xw = np.zeros(self.x.shape[1])
    self.reached_numeric_limit = False
    self.f_search = 0.
    self.n_search = 0.
    self.f_update = 0.
    self.cached_lambdas = np.zeros((2, 2))
    self.cached_lambdas[0, 0] = (self.x.dot(self.xs)).max() #max lambda necessary; guarantees M = 0
    self.cached_lambdas[0, 1] = np.inf
    self.cached_Ms = np.zeros(1, dtype=np.int64)

  def weights(self):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    return full_wts

  def error(self):
    return np.sqrt(((self.xw - self.xs)**2).sum())


