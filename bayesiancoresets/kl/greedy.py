import numpy as np
import warnings
from .kl import KLCoreset
from ..base.iterative import SingleGreedyCoreset
from ..base.optimization import adam


class GreedyKLCoreset(KLCoreset,SingleGreedyCoreset):

  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True):
    super().__init__(potentials=potentials, sampler=sampler, n_samples=n_samples, reverse=reverse, n_lognorm_disc=n_lognorm_disc, scaled=scaled, N=len(potentials))

  def _search(self):
    return self._kl_grad_estimate(self.wts, True).argmin()

  def _step_coeffs(self, f):
    def grd(x):
      #wnew = alpha*wold + beta 1n
      wi = x[0]*self.wts
      wi[f] += x[1]
      g = self._kl_grad_estimate(wi)
      ga = (self.wts*g).sum()
      gb = g[f]
      return np.array([ga, gb])
    ret= adam(np.array([1., 0.]), grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
    return ret[0], ret[1]
