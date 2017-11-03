import numpy as np

#TODO: split _compute_diam, _compute_normratio out into a utility func class

class FrankWolfe(object):
  def __init__(self, _x):
    x = np.asarray(_x)
    if len(x.shape) != 2 or not np.issubdtype(x.dtype, np.number):
      raise ValueError('FrankWolfe: input must be a 2d numeric ndarray')
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
    if self.x.size == 0:
      return 0.
    M = M if M else self.M
    if not self.diam:
      self._compute_diam()
    if not self.normratio:
      self._compute_normratio()
    if self.diam == 0. or self.normratio == 0.:
      return 0. if M > 0 else np.sqrt((self.xs**2).sum())
    if not self.nu:
      self._compute_nu()
    if self.nu == 0:
      return 0. if M > 0 else np.sqrt((self.xs**2).sum())
    lognum = np.log(self.sig) + np.log(self.normratio) + np.log(self.diam) + np.log(self.nu)
    logdenom_a = 2*np.log(self.diam) - 2.*(M-2.)*np.log(self.nu)
    if M > 1:
      logdenom_b = 2*np.log(self.normratio) + np.log(M-1.)
    else:
      logdenom_b = -np.inf 
    logdenom = 0.5*np.logaddexp(logdenom_a, logdenom_b)
    return np.exp(lognum - logdenom)
  
  def sqrt_bound(self, M=None):
    if self.x.size == 0:
      return 0.
    if not self.diam:
      self._compute_diam()
    return self.sig*self.diam/np.sqrt(M if M else self.M)

  def _compute_nu(self):
    from scipy.spatial import ConvexHull
    #reduce to low dimensional affine space if necessary
    vecs = self.sig*self.x/self.norms[:, np.newaxis]
    mvec = vecs.mean(axis=0)
    vecs -= mvec
    cov = vecs.T.dot(vecs) / vecs.shape[0]
    w, V = np.linalg.eigh(cov)
    V = V[:, w > 1e-8]
    vecs2 = vecs.dot(V)
    xs2 = (self.xs - mvec).dot(V)
    if V.shape[1] == 0:
      r = 0.
    elif V.shape[1] == 1:
      #if V is now 1-dimensional, compute the result directly
      rmin = (xs2 - vecs2).min()
      rmax = (xs2 - vecs2).max()
      if rmin > 0. or rmax < 0.:
        print 'FrankWolfe-- _compute_nu warning: rmin < 0 or rmax > 0. setting r to 0. rmin = ' + str(rmin) + ' rmax = ' + str(rmax)
        r = 0.
      else:
        r = min(np.fabs(rmin), np.fabs(rmax))
    else:
      #if V dim > 1, use Qhull
      #compute the half-space equations of the convex hull
      hull = ConvexHull(vecs2)
      b = hull.equations[:, -1]
      a = hull.equations[:, :-1]
      #for each half space constraint, the ball that touches it has radius (b -(aTx))/||a|| where x is the center
      #so take the minimum over all these radii
      r = ((b - a.dot(xs2))/np.sqrt((a**2).sum(axis=1))).min()
      if r < 0.:
        print 'FrankWolfe-- _compute_nu warning: r < 0 in compute_nu. setting to 0. r = ' + str(r)
        r = 0.
    self.nu = np.sqrt(max(0., 1. - r**2/(self.sig**2*self.diam**2))) 
    return

  def _compute_diam(self):
    if self.x.shape[0] == 1:
      self.diam = 0. 
      return
    normed_x = self.x/self.norms[:, np.newaxis] 
    distsqs = 2. - 2.*normed_x.dot(normed_x.T)
    self.diam = np.sqrt(max(0., distsqs.max()))
    return

  def _compute_normratio(self):
    if self.x.shape[0] == 1:
      self.normratio = 0. 
      return
    self.normratio = np.sqrt(max(0., 1. - (self.xs**2).sum()/self.sig**2))
    return




