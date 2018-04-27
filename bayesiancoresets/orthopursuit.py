import numpy as np
from .geometry import *
from scipy.optimize import minimize
import warnings
from .coreset import CoresetConstruction

class OrthoPursuit(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('FrankWolfe: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.norms = nrms[self.nzidcs]
    self.xs = self.x.sum(axis=0)
    self.N = self.x.shape[0]
    self.f_preproc = x.shape[0] + 2*self.x.shape[0] 
    self.f_search = 0.
    self.n_search = 0.
    self.f_update = 0.
    self.reached_numeric_limit = False
    self.prev_cost = np.sqrt((self.xs**2).sum())
    self.reset()

  #options are fast, accurate (fast tracks xw and wts separately, accurate updates xw from wts at each iter)
  def run(self, M):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('OrthoPursuit.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or self.reached_numeric_limit:
      warnings.warn('OrthoPursuit.run(): either data has no nonzero vectors or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    for m in range(self.M, M):
      #search for FW vertex and compute line search
      f = self.search()

      #check to make sure value to add is not in the current set (error should be ortho to current subspace)
      if self.wts[f] > 0:
        warnings.warn('OrthoPursuit.run(): search selected a nonzero weight to update')

      #run L-BFGS-B for optimal weight update
      self.wts[f] = 1e-6
      X = self.x[self.wts > 0, :]
      w0 = self.wts[self.wts > 0]
      res = minimize(fun = lambda w : ((self.xs - w.dot(X))**2).sum(), 
               x0 = w0, method='L-BFGS-B', 
               jac = lambda w : (w.dot(X)).dot(X.T) - 2*self.xs.dot(X.T), 
               bounds = [(0., None)]*w0.shape[0],
               options ={'ftol': 1e-12, 'gtol': 1e-9})
 
      #if the optimizer failed or our cost increased, stop
      if not res.success or np.sqrt(((self.xs - res.x.dot(X))**2).sum()) >= self.prev_cost:
        self.wts[f] = 0.
        self.reached_numeric_limit = True
        break

      #update weights, xw, and prev_cost
      self.wts[self.wts > 0] = res.x
      self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)
      self.prev_cost = self.error()
      self.f_update += (self.wts > 0).sum()  

      self.M = m+1

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
    self.prev_cost = self.error()
    self.reached_numeric_limit = False
    self.f_search = 0.
    self.n_search = 0.
    self.f_update = 0.

  def weights(self):
    #remap self.wts to the full original data size using nzidcs
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    return full_wts

  def error(self):
    return np.sqrt(((self.xw - self.xs)**2).sum())


class OrthoPursuit2(CoresetConstruction):

  def _xw_unscaled(self):
    return False
  
  def _initialize(self):
    self.prev_cost = np.sqrt((self.xs**2).sum())

  def _step(self, use_cached_xw):
    #search for FW vertex and compute line search
    f = self._search()

    #check to make sure value to add is not in the current set (error should be ortho to current subspace)
    if self.wts[f] > 0:
      warnings.warn(self.alg_name+'.run(): search selected a nonzero weight to update')

    #run L-BFGS-B for optimal weight update
    self.wts[f] = 1e-6
    X = self.x[self.wts > 0, :]
    w0 = self.wts[self.wts > 0]
    res = minimize(fun = lambda w : ((self.snorm*self.xs - w.dot(X))**2).sum(), 
             x0 = w0, method='L-BFGS-B', 
             jac = lambda w : (w.dot(X)).dot(X.T) - 2*self.snorm*self.xs.dot(X.T), 
             bounds = [(0., None)]*w0.shape[0],
             options ={'ftol': 1e-12, 'gtol': 1e-9})
 
    #if the optimizer failed or our cost increased, stop
    if not res.success or np.sqrt(((self.snorm*self.xs - res.x.dot(X))**2).sum()) >= self.prev_cost:
      self.wts[f] = 0.
      self.reached_numeric_limit = True
      return

    #update weights, xw, and prev_cost
    self.wts[self.wts > 0] = res.x
    self.xw = self.wts.dot(self.x)
    self.prev_cost = self.error()

  def _search(self):
    return (((self.snorm*self.xs - self.xw)*self.x).sum(axis=1)).argmax()


