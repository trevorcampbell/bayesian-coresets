import numpy as np
import warnings

def compute_nu(x, diam):
  from scipy.spatial import ConvexHull
  #if the diameter of the point set is 0, just return 0
  if diam == 0.:
    return 1., 0.

  #compute scaled data and sum
  mu = x.sum(axis=0)
  nrms = np.sqrt((x**2).sum(axis=1))
  v = nrms.sum()*x/nrms[:, np.newaxis]

  #shift data by the mean
  mv = v.mean(axis=0)
  v -= mv

  #scale all vectors down to have unit norm (for scaling cov numerical tolerance properly)
  vnrm = np.sqrt((v**2).sum(axis=1))
  vu = v.copy()
  vu[vnrm >0., :] /= vnrm[vnrm>0.][:, np.newaxis]
  vu -= vu.mean(axis=0)

  #get affine subspace the data lie in
  cov = vu.T.dot(vu) / vu.shape[0]
  w, W = np.linalg.eigh(cov)
  W = W[:, w > 1e-8]
  
  #project (if necessary)
  vP = v.dot(W)
  muP = (mu - mv).dot(W)

  if vP.shape[1] == 0:
    warnings.warn('geometry.compute_nu(): diam > 0 but affine subspace of data is dim 0. diam = ' + str(diam) + ' eigvals = ' + str(w))
  elif vP.shape[1] == 1:
    #if v is now 1-dimensional, compute the result directly
    rmin = (muP - vP).min()
    rmax = (muP - vP).max()
    if rmin > 0. or rmax < 0.:
      warnings.warn('geometry.compute_nu(): rmin < 0 or rmax > 0. setting r to 0. rmin = ' + str(rmin) + ' rmax = ' + str(rmax))
      r = 0.
    else:
      r = min(-rmin, rmax)
  else:
    #compute the half-space equations of the convex hull
    hull = ConvexHull(vP)
    b = hull.equations[:, -1]
    a = hull.equations[:, :-1]
    #for each half space constraint a^Tx <= b, the ball that touches it has radius (b -(aTx))/||a|| where x is the center
    #so take the minimum over all these radii
    #but ConvexHull is not guaranteed to output a particular normal for the hull, so we just take the fabs to automatically orient
    r = (np.fabs(b - a.dot(muP))/np.sqrt((a**2).sum(axis=1))).min()
  #output nu
  return np.sqrt(max(0., 1. - r**2/(nrms.sum()**2*diam**2))), r

def compute_diam(x):
  nrms = np.sqrt((x**2).sum(axis=1))
  #first normalize data
  v = x/nrms[:, np.newaxis] 
  #use dot product matrix to compute distsqs
  distsqs = 2. - 2.*v.dot(v.T)
  #rather than just output the max (2 - 2*(1-eps) isn't stable) get argmax and compute actual distsq from that
  n = distsqs.argmax()
  return np.sqrt( ((v[n/v.shape[0], :] - v[n % v.shape[0], :])**2).sum() )

def compute_normratio(x):
  s = (x.sum(axis=0)**2).sum()
  t = (np.sqrt((x**2).sum(axis=1))).sum()**2
  return np.sqrt(max(0., 1. - s/t))

#def compute_xi_tau(x):
#    xnrmsqs = (x**2).sum(axis=1)
#    distsqs = xnrmsqs[:, np.newaxis] + xnrmsqs - 2.*x.dot(x.T)
#    self.xi = self.x.shape[0]*np.sqrt(max(0., distsqs.max()))
#    self.tau = max(0., distsqs.sum())
#    return

