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
    #L = np.linalg.cholesky(H)

    lmb, V = np.linalg.eigh(H)
    if np.any(lmb<0):
      raise NumericalPrecisionError
    C = (V*np.sqrt(lmb)).T
    Cinv = (V/np.sqrt(lmb))

    B = C.dot(self.wts) - Cinv.T.dot(D)
    if self.update_single:
      A = np.atleast_2d(np.hstack((C.dot(one_f[:,np.newaxis]), C.dot(self.wts[:,np.newaxis]))))
    else:
      A = C
    w, resid = nnls(A,B) 
    gamma = self.step_sched(self.itrs)
    self._update(self.idcs, (1.-gamma)*self.wts + gamma*w)
    
    #lmb, V = np.linalg.eigh(HlogZa - H3logZa+1e-16*np.eye(HlogZa.shape[0]))
    #eta = 1.
    #while np.any(lmb <= 0.):
    #  eta /= 2.
    #  lmb, V = np.linalg.eigh(HlogZa - eta*H3logZa+1e-16*np.eye(HlogZa.shape[0]))
    #one_n = np.zeros(w.shape[0])
    #one_n[n] = 1.
    #C = (V*np.sqrt(lmb)).T
    #Cinv = (V/np.sqrt(lmb))
    #B = C.dot(w) + Cinv.T.dot(HlogZ1w)
    #if full:
    #  A = C
    #  w, resid = nnls(A,B) 
    #else:
    #  A = np.atleast_2d(np.hstack((C.dot(one_n[:,np.newaxis]), C.dot(w[:,np.newaxis]))))
    #  x, resid = nnls(A,B) 
    #  w = x[1]*w+x[0]*one_n
    #return w



    #if self.update_single:
    #  wtmp = self.wts.copy()
    #  wtmp[fidx] = 0.
    #  ab, resid = nnls(np.hstack((L.T.dot(wtmp)[:,np.newaxis], L.T.dot(onef)[:,np.newaxis])) , (L.T.dot(wtmp) - np.linalg.solve(L, D)))
    #  w = (ab[0]*wtmp + ab[1]*onef)
    #  self._update(self.idcs, (1.-gamma)*self.wts + gamma*)
    #  #ab, resid = nnls(np.hstack((L.T.dot(self.wts)[:,np.newaxis], L.T.dot(onef)[:,np.newaxis])) , (L.T.dot(self.wts) - np.linalg.solve(L, D)))
    #  #self._update(self.idcs, ab[0]*self.wts + ab[1]*onef)
    #else:
    #  w, resid = nnls(L.T, (L.T.dot(self.wts) - np.linalg.solve(L, D)))

    #gamma = self.step_sched(self.itrs)
    #self._update(self.idcs, (1.-gamma)*self.wts + gamma*w)
