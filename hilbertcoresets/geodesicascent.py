import numpy as np
import captree as ct
import warnings
import time
from gigasearch import GIGASearch

class GIGA(object):
  def __init__(self, _x): 
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('GIGA: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0] #number of vecs (incl zero vecs)
    self.x = x[self.nzidcs, :] #nonzero vectors

    self.norms = nrms[self.nzidcs] #norms of nonzero vecs
    self.sig = self.norms.sum() #norm sum
    self.xs = self.x.sum(axis=0) #sum of nonzero vecs (target vector)
    xsnrm = np.sqrt(((self.xs)**2).sum())
    self.y = self.x/self.norms[:, np.newaxis] #normalized nonzero data
    self.ys = self.xs/(xsnrm if xsnrm > 0. else 1.) #normalized sum vec
    self.N = self.x.shape[0] #number of nonzero vecs
    self.f_preproc = x.shape[0] + 3*self.x.shape[0] + 2
    self.f_update = 0.
    self.f_search_prev = 0.
    self.n_search_prev = 0.
    self.reached_numeric_limit = False
    self.search_module = GIGASearch(self.y)
    self.reset()

  #update_method can be 'fast' or 'accurate'
  #search_method can be 'exact', 'approximate'
  #if search_method is 'approximate', max_approx_search_evals should be set to some number << N
  #increasing the number improves the search but is more expensive
  def run(self, M, update_method='fast'):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('GIGA.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or (self.xs**2).sum() == 0. or self.reached_numeric_limit:
      warnings.warn('GIGA.run(): either data has no nonzero vectors, the sum has norm 0, or the numeric limit has been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    if self.y.shape[0] == 1:
      self.yw = self.y[0, :]
      self.wts[0] = 1.
      self.M = 1
      self.reached_numeric_limit = True
      self.f_update += 1.
      return
    if self.y.shape[1] == 1:
      self.yw = self.ys.copy()
      self.wts[np.argmax(self.y.dot(self.ys))] = 1.0
      self.M = 1
      self.reached_numeric_limit = True
      self.f_update += 1.
      return
      
    #this is commented out since initialization step is exactly the same as the main iteration if y(w) = 0
    #this is true for both tree and linear search
    #if self.M == 0:
    #  scores = self.y.dot(self.ys)
    #  f = scores.argmax()
    #  self.wts[f] = 1.
    #  self.yw = self.y[f, :]
    #  self.M = 1

    for m in range(self.M, M):
      f = self.search()
      gA = -1.
      gB = -1.
      if f >= 0:
        gA = self.ys.dot(self.y[f,:]) - self.ys.dot(self.yw) * self.yw.dot(self.y[f,:])
        gB = self.ys.dot(self.yw) - self.ys.dot(self.y[f,:]) * self.yw.dot(self.y[f,:])
        self.f_update += 3.
   
      #if the direction and/or line search failed
      if gA <= 0. or gB < 0:
        #try recomputing yw from scratch and rerunning search
        self.yw = (self.wts[:, np.newaxis]*self.y).sum(axis=0)
        nrm = np.sqrt((self.yw**2).sum())
        self.yw /= nrm
        self.wts /= nrm
        self.f_update += (self.wts > 0).sum() + 3.

        f = self.search()
        if f >= 0:
          gA = self.ys.dot(self.y[f,:]) - self.ys.dot(self.yw) * self.yw.dot(self.y[f,:])
          gB = self.ys.dot(self.yw) - self.ys.dot(self.y[f,:]) * self.yw.dot(self.y[f,:])
          self.f_update += 3.
        #if it still didn't work, we've reached the numeric limit
        if gA <= 0. or gB < 0:
          self.reached_numeric_limit = True
          break
      #direction+line search worked, update weights + yw
      gamma = gA/(gA+gB)
      self.wts *= (1.-gamma)
      self.wts[f] += gamma

      if update_method == 'fast':
        self.yw = (1.-gamma)*self.yw + gamma*self.y[f, :]
        self.f_update += 1.
      else:
        self.yw = (self.wts[:, np.newaxis]*self.y).sum(axis=0)
        self.f_update += (self.wts > 0).sum()

      nrm = np.sqrt((self.yw**2).sum())
      self.yw /= nrm
      self.wts /= nrm
      self.f_update += 3.
      self.M = m+1

    return

  def search(self):
    cdir = self.ys - self.ys.dot(self.yw)*self.yw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < 1e-14:
      return -1
    cdir /= cdirnrm
    return self.search_module.search(self.yw, cdir)
  
  #def search_linear(self):
  #  cdir = self.ys - self.ys.dot(self.yw)*self.yw
  #  cdirnrm =np.sqrt((cdir**2).sum()) 
  #  if cdirnrm < 1e-14:
  #    return -1
  #  cdir /= cdirnrm
  #  scorenums = self.y.dot(cdir) 
  #  scoredenoms = self.y.dot(self.yw)
  #  #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
  #  idcs = np.logical_and(scoredenoms > -1.+1e-14,  1.-scoredenoms**2 > 0.)
  #  #compute the norm 
  #  scoredenoms[idcs] = np.sqrt(1.-scoredenoms[idcs]**2)
  #  scoredenoms[np.logical_not(idcs)] = np.inf
  #  #compute the scores
  #  scores = scorenums/scoredenoms
  #  self.f_lin += 2.*self.N+2.
  #  return scores.argmax()
  
  #def search_tree(self, max_evals):

  #  if not self.tree.is_build_done():
  #    return self.search_linear()

  #  cdir = self.ys - self.ys.dot(self.yw)*self.yw
  #  cdirnrm =np.sqrt((cdir**2).sum()) 
  #  if cdirnrm < 1e-14:
  #    return -1
  #  cdir /= cdirnrm
  #  nopt = self.tree.search(self.yw, cdir, max_evals)
  #  return nopt
  
  #def build_tree(self):
  #  if self.tree_type == 'python':
  #    self.tree = ct.CapTree(self.y)
  #  else:
  #    self.tree = ct.CapTreeC(self.y)

  def get_num_ops(self):
    return self.f_preproc + self.f_update + self.search_module.num_search_ops() - self.f_search_prev 
 
  def get_num_nodes(self):
    return self.search_module.num_search_nodes() - self.n_search_prev

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.yw = np.zeros(self.y.shape[1])
    self.reached_numeric_limit = False
    self.f_search_prev = self.search_module.num_search_ops()
    self.n_search_prev = self.search_module.num_search_nodes()
    self.f_update = 0

  #options are fast, accurate (either use yw or recompute yw from wts)
  def error(self, method="fast"):
    #if M = 0, just output zeros using the fast method
    if self.M == 0:
      method = "fast"
    if method == "fast":
      return np.sqrt((self.xs**2).sum())*np.sqrt(max(0., 1. - self.yw.dot(self.ys)**2))
    else:
      ywt = (self.wts[:, np.newaxis]*self.y).sum(axis=0)
      ywtn = np.sqrt((ywt**2).sum())
      ywt /= ywtn
      w = ((self.wts/ywtn)/self.norms)*np.sqrt((self.xs**2).sum())*ywt.dot(self.ys)
      return np.sqrt((((w[:, np.newaxis]*self.x).sum(axis=0) - self.xs)**2).sum())

  #options are accurate and fast (either use yw or recompute)
  #by default use accurate computation for weights
  def weights(self, method="accurate"):
    #if M = 0, just output zeros using the fast method
    if self.M == 0:
      method = "fast"
    full_wts = np.zeros(self.full_N)
    if method == "fast":
      full_wts[self.nzidcs] = (self.wts/self.norms)*np.sqrt((self.xs**2).sum())*self.yw.dot(self.ys)
    else:
      ywt = (self.wts[:, np.newaxis]*self.y).sum(axis=0)
      ywtn = np.sqrt((ywt**2).sum())
      ywt /= ywtn
      full_wts[self.nzidcs] = ((self.wts/ywtn)/self.norms)*np.sqrt((self.xs**2).sum())*ywt.dot(self.ys)
    return full_wts

  def exp_bound(self, M=None):
    raise NotImplementedError("GIGA.exp_bound(): not implemented")

  def sqrt_bound(self, M=None):
    raise NotImplementedError("GIGA.sqrt_bound(): not implemented")
    
