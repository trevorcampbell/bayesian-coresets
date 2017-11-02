import numpy as np

#TODO: split _compute_diam, _compute_normratio out?
#TODO: _compute_nu assumes full rank data Gram matrix; fix that by projection of some sort

class FrankWolfe(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2:
      raise ValueError('FrankWolfe: input is not a 2d ndarray')
    nrms = np.sqrt((x**2).sum(axis=1))
    self.nzidcs = nrms > 0.
    self.full_N = x.shape[0]
    self.x = x[self.nzidcs, :]

    self.norms = nrms[self.nzidcs]
    self.sig = self.norms.sum()
    self.xs = self.x.sum(axis=0)
    self.N = self.x.shape[0]
    self.normratio = None
    self.diam = None
    self.nu = None
    self.reset()

  def run(self, M, update_method='fast'):
    if self.x.size == 0:
      return
    if M <= self.M:
      print 'Warning: requested M <= self.M, returning without modifying weights'
      return

    if self.M == 0:
      scores = (self.xs*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      self.wts[f] = self.sig/self.norms[f]
      self.xw = self.sig/self.norms[f]*self.x[f, :]
      self.M = 1

    for m in range(self.M, M):
      scores = ((self.xs - self.xw)*self.x).sum(axis=1)/self.norms
      f = scores.argmax()
      gammanum = (self.sig/self.norms[f]*self.x[f, :] - self.xw).dot(self.xs - self.xw)
      gammadenom = (self.sig/self.norms[f]*self.x[f, :] - self.xw).dot(self.sig/self.norms[f]*self.x[f, :] - self.xw) 
      gamma = gammanum/gammadenom
      if gamma < 0 or gamma > 1:
        print 'Warning: gamma not in [0, 1]: ' + str(gamma)
        gamma = (0. if gamma < 0 else gamma)
        gamma = (1. if gamma > 1 else gamma)
      self.wts *= (1.-gamma)
      self.wts[f] += gamma*self.sig/self.norms[f] 
      if update_method == 'fast':
        self.xw = (1.-gamma)*self.xw + gamma*self.sig/self.norms[f]*self.x[f, :]
      else:
        self.xw = (self.wts[:, np.newaxis]*self.x).sum(axis=0)

    self.M = M
    return

  def reset(self):
    self.M = 0
    self.wts = np.zeros(self.N)
    self.xw = np.zeros(self.x.shape[1])

  def weights(self):
    full_wts = np.zeros(self.full_N)
    full_wts[self.nzidcs] = self.wts
    return full_wts

  def error(self):
    return np.sqrt(((self.xw - self.xs)**2).sum())

  def exp_bound(self, M=None):
    if not self.diam:
      self._compute_diam()
    if not self.nu:
      self._compute_nu()
    if not self.normratio:
      self._compute_normratio()
    M = M if M else self.M
    return self.sig*normratio*self.diam*self.nu/np.sqrt(self.diam**2*np.power(self.nu, -2.*(M-2.)) + normratio**2*(M-1))
  
  def sqrt_bound(self, M=None):
    if not self.diam:
      self._compute_diam()
    return self.sig*self.diam/np.sqrt(M if M else self.M)

  def _compute_nu(self):
    from scipy.spatial import ConvexHull
    #compute the half-space equations of the convex hull
    hull = ConvexHull(self.sig*self.x/self.norms[:, np.newaxis])
    b = hull.equations[:, -1]
    a = hull.equations[:, :-1]
    #for each half space constraint, the ball that touches it has radius (b -(aTx))/||a|| where x is the center
    #so take the minimum over all these radii
    r = ((b - a.dot(self.xs))/np.sqrt((a**2).sum(axis=1))).min()
    self.nu = np.sqrt(1. - r**2/(self.sig**2*self.diam**2)) 
    return

  def _compute_diam(self):
    normed_x = self.x/self.norms[:, np.newaxis] 
    distsqs = 2. - 2.*normed_x.dot(normed_x.T)
    self.diam = np.sqrt(distsqs.max())
    return

  def _compute_normratio(self):
    self.normratio = np.sqrt(1. - (self.xs**2).sum()/self.sig**2)
    return




