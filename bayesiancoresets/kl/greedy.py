import numpy as np
import warnings
from .kl import KLCoreset
from ..base.iterative import SingleGreedyCoreset
from ..base.optimization import adam


class GreedyKLCoreset(KLCoreset, SingleGreedyCoreset):

  def _search(self):
    #print('search result: ' + str(self._kl_grad(self.wts, True).argmin()))
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
    ret = adam(np.array([1., 0.]), grd, opt_itrs=self.opt_itrs, adam_a1=self.adam_a1, adam_a2=self.adam_a2)
    return ret[0], ret[1]
