import numpy as np
from ..base.incremental import IncrementalCoreset
from ..util.opt import nn_opt
from .kl import KLCoreset

class SparseVICoreset(KLCoreset,IncrementalCoreset):

  def __init__(self, N, tangent_space_factory, step_sched = lambda i : 1./(1.+i), opt_itrs=1000, update_single = True):
    super().__init__(N=N) 
    self.tsf = tangent_space_factory
    self.update_single = update_single
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs

  def _select(self):
    #construct a new tangent space for this search iteration
    T = self.tsf(self.wts, self.idcs)
    #compute the correlations
    corrs = T.kl_residual_correlations()
    #for any in the active set, just look at corr mag
    corrs[self.idcs] = np.fabs(corrs[self.idcs]) 
    return np.argmax(corrs)

  def _reweight(self, f):
    if f not in self.idcs:
      self._update(f, 0.)
    fidx = np.where(self.idcs == f)[0][0]

    onef = np.zeros(self.idcs.shape[0])
    onef[fidx] = 1.
   
    if self.update_single:
      wtmp = self.wts.copy()
      wtmp[fidx] = 0.
      #since below uses nn_opt, will parametrize beta w + alpha 1_n with w[fidx] = 0
      x0 = np.array([1., self.wts[fidx]]) #scale, amt of new wt
      def grd(ab):
        T = self.tsf(ab[0]*wtmp + ab[1]*onef, self.idcs)
        g = T.kl_grad(grad_idcs=self.idcs)
        ga = wtmp.dot(g)
        gb = g[fidx]
        return np.array([ga, gb])
      x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
      self._update(self.idcs, x[0]*wtmp + x[1]*onef)

      #closer to notation from paper (but since nn_opt, wrong constraints)
      #x0 = np.array([1., 0.]) #scale, amt of new wt
      #def grd(ab):
      #  T = self.tsf(ab[0]*self.wts + ab[1]*onef, self.idcs)
      #  g = T.kl_grad(grad_idcs=self.idcs)
      #  ga = self.wts.dot(g)
      #  gb = g[fidx]
      #  return np.array([ga, gb])
      #x = nn_opt(x0, grd, opt_itrs=1000, step_sched = self.step_sched)
      #self._update(self.idcs, x[0]*self.wts + x[1]*onef)
    else:
      x0 = self.wts
      def grd(w):
        T = self.tsf(w, self.idcs)
        g = T.kl_grad(grad_idcs=self.idcs)
        return g
      x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
      self._update(self.idcs, x)
    return

