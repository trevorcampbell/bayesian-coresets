from .coreset import GreedySingleUpdate
import numpy as np

class MatchingPursuit(GreedySingleUpdate):

  def _xw_unscaled(self):
    return False

  def _search(self):
    res = self.snorm*self.xs - self.xw
    dots = (res*self.x).sum(axis=1)
    widcs = np.argwhere(self.wts > 0).flatten()

    #if no active indices, just output argmax
    if widcs.shape[0] == 0:
      return dots.argmax()
    
    #search positive direction on whole dataset, negative direction on active set
    fpos = dots.argmax()
    pos = dots[fpos]
    fneg = (-dots[widcs]).argmax()
    neg = (-dots[widcs])[fneg]

    if pos >= neg:
      return fpos
    else:
      return widcs[fneg]

  def _step_coeffs(self, f):
    alpha = 1.0
    beta = (self.snorm*self.xs - self.xw).dot(self.x[f,:])
    if beta < -self.wts[f]:
      beta = -self.wts[f]
    return alpha, beta


#   old reweighting code
#    xwsq = ((self.xw)**2).sum()
#    xwxn = self.xw.dot(self.x[f, :])
#    det = xwsq - xwxn**2
#
#    #check the two principal minors of the cost matrix
#    if xwsq > 0 and det > 1e-9:
#      #if ||x(w)|| > 0 and the matrix is reasonably well-conditioned, use closed-form soln
#      xxw = (self.snorm*self.xs).dot(self.xw)
#      xxn = (self.snorm*self.xs).dot(self.x[f, :])
#      
#      alpha = (1./det)*(xxw - xwxn*xxn)
#      beta = (1./det)*(xwsq*xxn - xwxn*xxw)
#    elif xwsq == 0:
#      #again use closed-form if x(w) = 0
#      alpha = 0
#      beta = (self.snorm*self.xs).dot(self.x[f, :])
#    else:
#      #in the worst case (ill-conditioned matrix), use a nonnegative lsq solver
#      A = np.vstack((self.xw, self.x[f, :])).T
#      res = lsq_linear(A, self.snorm*self.xs, bounds=(0., np.inf))
#      #if the optimizer failed or our cost increased, stop
#      prev_cost = self.error()
#      if not res.success or np.sqrt(2.*res.cost) >= prev_cost:
#        return None, None
#      alpha = res.x[0]
#      beta = res.x[1]
#
#    if beta < 0. or alpha < 0. or (beta == 0. and alpha == 1.):
#      return None, None
#    return alpha,  beta

  #def _step_coeffs(self, f):
    #Old code
    #v1 = self.xw - self.xw.dot(self.x[f, :])*self.x[f, :]
    #v2 = (self.xw**2).sum()*self.x[f, :] - self.xw.dot(self.x[f, :])*self.xw
   
    #if (v1**2).sum() == 0.:
    #  return None, None
    #
    #alpha = (self.snorm*self.xs).dot(v1) / self.xw.dot(v1)
    #beta = (self.snorm*self.xs).dot(v2) / self.xw.dot(v1)

    #if beta < 0. or alpha < 0. or (beta == 0. and alpha == 1.):
    #  return None, None
    #return alpha,  beta


  #def _initialize(self):
  #  f = self._search()
  #  self.wts[f] = self.snorm*self.x[f, :].dot(self.xs)
  #  self.xw = self.wts[f]*self.x[f, :]
  #  self.M = 1

