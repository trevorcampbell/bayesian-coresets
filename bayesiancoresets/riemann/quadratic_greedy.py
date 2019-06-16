import numpy as np
from ..base.iterative import GreedyCoreset
from scipy.optimize import nnls
from .kl import KLCoreset
from ..util.errors import NumericalPrecisionError 

class QuadraticSparseVICoreset(KLCoreset,GreedyCoreset):

  def __init__(self, N, tangent_space_factory, step_sched = lambda i : np.sqrt(1./(1.+i)), update_single = True):
    super().__init__(N=N) 
    self.tsf = tangent_space_factory
    self.update_single = update_single
    self.step_sched = step_sched

  def _search(self):
    #construct a new tangent space for this search iteration
    T = self.tsf(self.wts, self.idcs)
    #compute the correlations
    corrs = T.kl_residual_correlations()
    #for any in the active set, just look at corr mag
    corrs[self.idcs] = np.fabs(corrs[self.idcs]) 
    return np.argmax(corrs)

  def _update_weights(self, f):
    if f not in self.idcs:
      self._update(f, 0.)
    fidx = np.where(self.idcs == f)[0][0]

    onef = np.zeros(self.idcs.shape[0])
    onef[fidx] = 1.

    T = self.tsf(self.wts, self.idcs)
    D, H = T.kl_quadratic_expansion(self.idcs)

    lmb, V = np.linalg.eigh(H)
    if np.any(lmb<0):
      raise NumericalPrecisionError
    C = (V*np.sqrt(lmb)).T
    Cinv = (V/np.sqrt(lmb))

    B = C.dot(self.wts) - Cinv.T.dot(D)
    if self.update_single:
      wtmp = self.wts.copy()
      wtmp[fidx] = 0.
      A = np.atleast_2d(np.hstack((C.dot(onef[:,np.newaxis]), C.dot(wtmp[:,np.newaxis]))))
      ab, resid = nnls(A,B) 
      w = ab[0]*onef+ab[1]*wtmp
    else:
      A = C
      w, resid = nnls(A,B) 
    gamma = self.step_sched(self.itrs)
    self._update(self.idcs, (1.-gamma)*self.wts + gamma*w)
   
