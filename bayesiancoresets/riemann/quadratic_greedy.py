import numpy as np
from ..base.iterative import GreedyCoreset
from scipy.optimize import nnls
from .kl import KLCoreset

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
    D, H = T.kl_quadratic_expansion()
    L = np.linalg.cholesky(H)

    gamma = self.step_sched(self.itrs)
    if self.update_single:
      wtmp = self.wts.copy()
      wtmp[fidx] = 0.
      ab, resid = nnls(np.hstack((L.T.dot(wtmp)[:,np.newaxis], L.T.dot(onef)[:,np.newaxis])) , (L.T.dot(wtmp) - np.linalg.solve(L, D)))
      self._update(self.idcs, (1.-gamma)*self.wts + gamma*(ab[0]*wtmp + ab[1]*onef))
      #ab, resid = nnls(np.hstack((L.T.dot(self.wts)[:,np.newaxis], L.T.dot(onef)[:,np.newaxis])) , (L.T.dot(self.wts) - np.linalg.solve(L, D)))
      #self._update(self.idcs, ab[0]*self.wts + ab[1]*onef)
    else:
      w, resid = nnls(L.T, (L.T.dot(self.wts) - np.linalg.solve(L, D)))
      self._update(self.idcs, (1.-gamma)*self.wts + gamma*w)
