import numpy as np
import captree as ct
import warnings

class GIGA(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('GIGA: input must be a 2d numeric ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.norms = nrms[self.nzidcs]
    self.sig = self.norms.sum()
    self.xs = self.x.sum(axis=0)
    self.y = self.x/self.norms[:, np.newaxis]
    self.ys = self.xs/np.sqrt(((self.xs)**2).sum())
    self.N = self.x.shape[0]
    self.tree = None
    self.f_tree = 0.
    self.m_tree = 0.
    self.s_tree = 0.
    self.n_tree = 0.
    self.f_lin = 0.
    #self.m_lin = 0.
    #self.s_lin = 0.
    self.n_lin = 0.
    self.reached_numeric_limit = False
    self.reset()

  #update_method can be 'fast' or 'stable'
  #search_method can be 'adaptive', 'linear', or 'tree'
  def run(self, M, update_method='fast', search_method='adaptive', M_max = None):
    #if M is not greater than self.M, just return 
    if M <= self.M:
      raise ValueError('GIGA.run(): M must be increasing. self.M = '+str(self.M) + ' M = '+str(M))
    if self.x.size == 0 or (self.xs**2).sum() == 0. or self.reached_numeric_limit:
      warnings.warn('GIGA.run(): either data has no nonzero vectors, the sum has norm 0, or the numeric limit has \ 
                     been reached. No more iterations will be run. M = ' + str(self.M) + ', error = ' +str(self.error()))
      return

    if self.y.shape[0] == 1:
      self.yw = self.y[0, :]
      self.wts[0] = 1.
      self.M = 1
      self.reached_numeric_limit = True
      return
    if self.y.shape[1] == 1:
      self.yw = self.ys.copy()
      self.wts[np.argmax(self.ys.dot(self.y))] = 1.0
      self.M = 1
      self.reached_numeric_limit = True
      return
      

    if search_method == 'linear':
      GIGA.search = GIGA.search_linear
    elif search_method == 'tree':
      GIGA.search = GIGA.search_tree
      if not self.tree:
        self.build_tree()
    else:
      if not M_max:
        M_max = M
      if self.tree:
        GIGA.search = GIGA.search_adaptive
      elif not self.tree and M_max - self.M >= (4*self.N+3)*np.log2(self.N)/(2.*(self.N+1-np.log2(self.N))):
        GIGA.search = GIGA.search_adaptive
        self.build_tree()
      else:
        GIGA.search = GIGA.search_linear

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
      if f:
        gA = self.ys.dot(self.y[f,:]) - self.ys.dot(self.yw) * self.yw.dot(self.y[f,:])
        gB = self.ys.dot(self.yw) - self.ys.dot(self.y[f,:]) * self.yw.dot(self.y[f,:])
   
      #if the direction and/or line search failed
      if gA <= 0. or gB < 0:
        #try recomputing yw from scratch and rerunning search
        self.yw = (self.wts[:, np.newaxis]*self.y).sum(axis=0)
        nrm = np.sqrt((self.yw**2).sum())
        self.yw /= nrm
        self.wts /= nrm

        f = self.search()
        if f:
          gA = self.ys.dot(self.y[f,:]) - self.ys.dot(self.yw) * self.yw.dot(self.y[f,:])
          gB = self.ys.dot(self.yw) - self.ys.dot(self.y[f,:]) * self.yw.dot(self.y[f,:])
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
      else:
        self.yw = (self.wts[:, np.newaxis]*self.y).sum(axis=0)

      nrm = np.sqrt((self.yw**2).sum())
      self.yw /= nrm
      self.wts /= nrm
      self.M = m+1

    return

  def search_linear(self):
    cdir = self.ys - self.ys.dot(self.yw)*self.yw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm == 0.:
      return None
    cdir /= cdirnrm
    scorenums = self.y.dot(cdir) 
    scoredenoms = 1.-self.y.dot(self.yw)**2
    scoredenoms[scoredenoms < 1e-12] = np.inf
    scores = scorenums/np.sqrt(scoredenoms)
    #cdir /= np.sqrt((cdir**2).sum())
    #dirs = self.y - self.y.dot(self.yw)[:,np.newaxis]*self.yw
    #dirnrms = np.sqrt((dirs**2).sum(axis=1))
    #dirnrms[dirnrms < 1e-16] = np.inf #this is only really used in iteration M=2, where yw = the initial vector
    #dirs /= dirnrms[:, np.newaxis]
    #scores = dirs.dot(cdir)
    self.f_lin += 2.*self.N+2.
    #self.s_lin += np.log(2.*self.N+2.)**2
    #self.m_lin += np.log(2.*self.N+2.)
    self.n_lin += 1
    return scores.argmax()
  
  def search_tree(self):
    cdir = self.ys - self.ys.dot(self.yw)*self.yw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm == 0.:
      return None
    cdir /= cdirnrm
    nopt, nfun = ct.cap_tree_search(self.tree, self.yw, cdir)
    self.f_tree += nfun
    self.m_tree += np.log(nfun)
    self.s_tree += np.log(nfun)**2
    self.n_tree += 1
    return nopt
  
  def search_adaptive(self):
    #this uses UCB1-Normal from Auer et al ``Finite-time Analysis of the Multiarmed Bandit Problem'' (2002)
    #modification: since we know linear search is 2N+2 ops, dont need confidence bounds for that
    n = self.n_tree+self.n_lin + 1
    #if self.n_lin < 2 or self.n_lin < np.ceil(8.*np.log(n)):
    #  return self.search_linear()
    if self.n_tree < 2 or self.n_tree < np.ceil(8.*np.log(n)):
      return self.search_tree()
    #lin_idx = self.m_lin/self.n_lin + np.sqrt(16.*((self.s_lin - self.m_lin**2/self.n_lin)/(self.n_lin-1))*(np.log(n-1.)/self.n_lin))
    tree_idx = self.m_tree/self.n_tree - np.sqrt(16.*(max(0., self.s_tree - self.m_tree**2/self.n_tree)/(self.n_tree-1))*(np.log(n-1.)/self.n_tree))
    if tree_idx < np.log(2.*self.N+2):
      return self.search_tree()
    else:
      return self.search_linear()

  def build_tree(self):
    self.tree = ct.CapTree(self.y)

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.yw = np.zeros(self.y.shape[1])

  def error(self):
    return np.sqrt((self.xs**2).sum())*np.sqrt(max(0., 1. - self.yw.dot(self.ys)**2))

  def weights(self):
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = (self.wts/self.norms)*np.sqrt((self.xs**2).sum())*self.yw.dot(self.ys)
    return full_wts


  ########
  ########
  #THE BELOW ARE PLACEHOLDERS DESIGNED TO PASS TESTS; THESE NEED TO BE IMPLEMENTED
  ########
  ########

  def exp_bound(self, M=None):
    if self.x.size == 0:
      return 0.
    M = np.floor(M) if M else self.M
    if M <= 0:
      raise ValueError('GIGA.exp_bound(): M must be >= 1. Requested M: '+str(M))
    if M > 1e99:
      return 0.
    return np.iinfo(np.int64).max - M #TODO

  def sqrt_bound(self, M=None):
    if self.x.size == 0:
      return 0.
    M = np.floor(M) if M else self.M
    if M <= 0:
      raise ValueError('GIGA.exp_bound(): M must be >= 1. Requested M: '+str(M))
    if M > 1e99:
      return 0.
    return np.iinfo(np.int64).max - M #TODO

