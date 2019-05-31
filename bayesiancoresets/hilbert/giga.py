import numpy as np
from ..base.iterative import GreedySingleUpdateCoreset
from ..base.errors import NumericalPrecisionError
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
    nw = self.T.sum_w_norm()
    xs = self.T.sum()
    ns = self.T.sum_norm()

    nw = 1. if nw == 0. else nw
    if ns == 0.:
      raise NumericalPrecisionError

    xw /= nw
    xs /= ns

    cdir = xs - xs.dot(xw)*xw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < TOL:
      raise NumericalPrecisionError
    cdir /= cdirnrm
    scorends = (self.T[:]/self.T.norms()[:,np.newaxis]).dot(np.hstack((cdir[:,np.newaxis], xw))) 
    #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
    idcs = np.logical_and(scoredns[:,1] > -1.+1e-14,  1.-scoredns[:,1]**2 > 0.)
    #compute the norm 
    scoredns[idcs, 1] = np.sqrt(1.-scoredns[idcs,1]**2)
    scoredns[np.logical_not(idcs),1] = np.inf
    #compute the scores and argmax
    return (scoredns[:,0]/scoredns[:,1]).argmax()
 
  def _step_coeffs(self, f):
    xw = self.T.sum_w(self.wts, self.idcs)
    nw = self.T.sum_w_norm()
    xs = self.T.sum()
    ns = self.T.sum_norm()
    xf = self.T[f]
    nf = self.T.norms()[f]

    xw /= nw
    xs /= ns
    xf /= nf

    gA = xs.dot(xf) - xs.dot(xw) * xw.dot(xf)
    gB = xs.dot(xw) - xs.dot(xf) * xw.dot(xf)
    if gA <= 0. or gB < 0:
      raise NumericalPrecisionError
    return gB/(gA+gB), gA/(gA+gB) 


