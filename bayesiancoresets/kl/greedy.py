import numpy as np
import warnings
from .kl import KLCoreset
from ..base.iterative import GreedySingleUpdateCoreset
from ..base.optimization import adam


class GreedyKLCoreset(KLCoreset,GreedySingleUpdateCoreset):

  def _search(self):
    return self._kl_grad_estimate(self.wts).argmin()

  def _step_coeffs(self, f):
    def grd(x):
      #wnew = alpha*wold + beta 1n
      wi = x[0]*self.wts
      wi[f] += x[1]
      g = self._kl_grad_estimate(wi)
      ga = (self.wts*g).sum()
      gb = g[f]
      return np.array([ga, gb])
    return adam(np.array([1., 0.]), grd, opt_itrs=1000, adam_a=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
