import numpy as np
import warnings
from .kl import KLCoreset
from ..base.iterative import SingleGreedyCoreset
from ..base.optimization import adam


class GreedyKLCoreset(KLCoreset, SingleGreedyCoreset):

  def _search(self):
    return self._kl_grad(self.wts, True).argmin()

  def _step_coeffs(self, f):
    def grd(x):
      #wnew = alpha*wold + beta 1n
      wi = x[0]*self.wts
      wi[f] += x[1]
      g = self._kl_grad(wi)
      ga = (self.wts*g).sum()
      gb = g[f]
      return np.array([ga, gb])
    ret= adam(np.array([1., 0.]), grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
    return ret[0], ret[1]
