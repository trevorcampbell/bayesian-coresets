import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt

class KLCoreset(object):
  def _optimize(self):
    x0 = self.wts
      def grd(w):
        T = self.tsf(w, self.idcs)
        g = T.kl_grad(grad_idcs=self.idcs)
        return g
      x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
      self._update(self.idcs, x)

  def error(self):
    return 0.
