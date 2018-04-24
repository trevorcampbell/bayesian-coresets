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
    self.cached_lambdas = np.zeros((2, 2))
    self.cached_lambdas[0, 0] = (self.x.dot(self.xs)).max() #max lambda necessary; guarantees M = 0
    self.cached_lambdas[0, 1] = np.inf
    self.cached_Ms = np.zeros(2, dtype=np.int64)
    self.cached_Ms[1] = np.inf
    self.reset()

  def run(self, M, tol=1e-9):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('FrankWolfe.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or self.reached_numeric_limit:
      warnings.warn('FrankWolfe.run(): either data has no nonzero vectors or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    # run bisection search to find lambda value that yields ||w||_0 = M
    # ||w||_0 is a decreasing function in lambda, with ||w||_0 = 0 when lambda >= lambda_max
    
    #first, get cached lambda bounds / wts for M
    Midx = np.searchsorted(self.cached_Ms, M, side='right')
    lmb_l = self.cached_lambdas[Midx, 1]
    lmb_r = self.cached_lambdas[Midx-1, 0]
    
    self.M = self.cached_Ms[Midx-1]

    #bisection search
    while self.M != M:
      lmb = 0.5*(lmb_l + lmb_r)
      #TODO: run cycling descent with lmb until convergence; outputs self.M and self.wts
      #lasso cycling coordinate descent update
      #w_j =  max( (r_j^Tx_j - lambda) / ||x_j||^2, 0)
      #r_j = y - X_{-j}^Tw_{-j}
      #self.M = ...
      #self.wts = ...
      
      #cache the result if we might need it in the future
      if self.M >= M:
        lmb_l = lmb
        Midx = np.searchsorted(self.cached_Ms, self.M)
        if self.cached_Ms[Midx] == self.M:
          self.cached_lambdas[Midx, 0] = min(lmb, self.cached_lambdas[Midx, 0])
          self.cached_lambdas[Midx, 1] = max(lmb, self.cached_lambdas[Midx, 1])
        else:
          np.insert(self.cached_Ms, Midx, self.M)
          np.insert(self.cached_lambdas, Midx, np.ones(2)*lmb)
      else:
        lmb_r = lmb
      
    #get rid of all cached values for Ms less than self.M (since M has to always increase)
    keep_idcs = self.cached_Ms >= self.M
    self.cached_Ms = self.cached_Ms[keep_idcs]
    self.cached_lambdas = self.cached_lambdas[keep_idcs, :] 

    #run l-bfgs-b to get optimal weights in support
    #TODO
    
    #set xw and wts

    return

  def search(self):
    scores = ((self.xs - self.xw)*self.x).sum(axis=1)/self.norms
    self.f_search += 2. + self.x.shape[0]
    self.n_search += self.x.shape[0]
    return scores.argmax()

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

  #options are standard (just output the FW weights), scaled (apply the optimal scaling first)
  def weights(self, method="standard"):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    if method == "standard":
      return full_wts
    else:
      norm_xs = np.sqrt((self.xs**2).sum())
      norm_xw = np.sqrt((self.xw**2).sum())
      dotws = (self.xw*self.xs).sum()
      return full_wts*(norm_xs/norm_xw)*max(0., dotws/(norm_xs*norm_xw))

  #options are fast, accurate (either use xw or recompute xw from wts)
  def error(self, method="fast"):
    if method == "fast":
      return np.sqrt(((self.xw - self.xs)**2).sum())
    else:
      return np.sqrt((((self.wts[:, np.newaxis]*self.x).sum(axis=0) - self.xs)**2).sum())


