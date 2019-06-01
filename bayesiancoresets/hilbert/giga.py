import numpy as np
from ..base.iterative import GreedySingleUpdateCoreset
from ..util.errors import NumericalPrecisionError
from .. import TOL

class GIGACoreset(GreedySingleUpdateCoreset):

  def __init__(self, tangent_space):
    super().__init__(N=tangent_space.num_vectors()) 
    self.T = tangent_space
    if np.any(self.T.norms() == 0):
      raise ValueError(self.alg_name+'.__init__(): tangent space must not have any 0 vectors')

  def error(self):
    return self.T.error(self.wts, self.idcs)

  def _search(self):
    xw = self.T.sum_w(self.wts, self.idcs)
    nw = self.T.sum_w_norm(self.wts, self.idcs)
    xs = self.T.sum()
    ns = self.T.sum_norm()

    nw = 1. if nw == 0. else nw
    if ns == 0.:
      raise NumericalPrecisionError('norm of sum = 0')

    cdir = xs/ns - (xs/ns).dot((xw/nw))*(xw/nw)
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < TOL:
      raise NumericalPrecisionError('cdirnrm < TOL: cdirnrm = ' + str(cdirnrm))
    cdir /= cdirnrm
    scorends = (self.T[:]/self.T.norms()[:,np.newaxis]).dot(np.hstack((cdir[:,np.newaxis], xw[:,np.newaxis]/nw))) 
    #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
    idcs = np.logical_and(scorends[:,1] > -1.+1e-14,  1.-scorends[:,1]**2 > 0.)
    #compute the norm 
    scorends[idcs, 1] = np.sqrt(1.-scorends[idcs,1]**2)
    scorends[np.logical_not(idcs),1] = np.inf
    #compute the scores and argmax

    return (scorends[:,0]/scorends[:,1]).argmax()
 
  def _step_coeffs(self, f):
    xw = self.T.sum_w(self.wts, self.idcs)
    nw = self.T.sum_w_norm(self.wts, self.idcs)
    xs = self.T.sum()
    ns = self.T.sum_norm()
    xf = self.T[f]
    nf = self.T.norms()[f]

    nw = 1. if nw == 0. else nw

    gA = (xs/ns).dot((xf/nf)) - (xs/ns).dot((xw/nw)) * (xw/nw).dot((xf/nf))
    gB = (xs/ns).dot((xw/nw)) - (xs/ns).dot((xf/nf)) * (xw/nw).dot((xf/nf))
    if gA <= 0. or gB < 0:
      raise NumericalPrecisionError

    a = gB/(gA+gB)/nw
    b = gA/(gA+gB)/nf
    
    x = a*xw + b*xf
    nx = np.sqrt((x**2).sum())
    scale = ns/nx*(x/nx).dot(xs/ns)
    
    return a*scale, b*scale


