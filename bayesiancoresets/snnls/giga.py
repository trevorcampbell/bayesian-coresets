import numpy as np
from ..util.errors import NumericalPrecisionError
from .. import util
from .snnls import SparseNNLS

class GIGA(SparseNNLS):

  def __init__(self, A, b):
    super().__init__(A, b)

    Anorms = np.sqrt((self.A**2).sum(axis=0))
    if np.any( Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')
    self.An = self.A / Anorms

    self.bnorm = np.sqrt(((self.b)**2).sum())
    if self.bnorm == 0.:
      raise NumericalPrecisionError('norm of b must be > 0')
    self.bn = self.b / self.bnorm

  def _select(self):
    xw = self.A.dot(self.w)
    nw = np.sqrt(((xw)**2).sum())
    nw = 1. if nw == 0. else nw
    xw /= nw

    cdir = self.bn - self.bn.dot(xw)*xw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < util.TOL:
      raise NumericalPrecisionError('cdirnrm < TOL: cdirnrm = ' + str(cdirnrm))
    cdir /= cdirnrm
    scorends = self.An.T.dot(np.hstack((cdir[:,np.newaxis], xw[:,np.newaxis]))) 
    #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
    idcs = np.logical_and(scorends[:,1] > -1.+1e-14,  1.-scorends[:,1]**2 > 0.)
    #compute the norm 
    scorends[idcs, 1] = np.sqrt(1.-scorends[idcs,1]**2)
    scorends[np.logical_not(idcs),1] = np.inf
    #compute the scores and argmax
    return (scorends[:,0]/scorends[:,1]).argmax()
 
  def _step_coeffs(self, f):

    #TODO fix this

    xw = self.A.dot(self.w)
    nw = np.sqrt(((xw)**2).sum())
    nw = 1. if nw == 0. else nw
    xw /= nw

    xf = self.A[:, f]
    nf = np.sqrt((xf**2).sum())
    xf /= nf

    gA = self.bn.dot(xf) - self.bn.dot(xw) * xw.dot(xf)
    gB = self.bn.dot(xw) - self.bn.dot(xf) * xw.dot(xf)
    if gA <= 0. or gB < 0:
      raise NumericalPrecisionError

    a = gB/(gA+gB)
    b = gA/(gA+gB)
    
    x = a*xw + b*xf
    nx = np.sqrt((x**2).sum())
    scale = self.bnorm/nx*(x/nx).dot(xs/ns)
    
    return a*scale, b*scale


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



