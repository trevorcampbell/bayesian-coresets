import numpy as np
import warnings
from .kl import KLCoreset
from ..base.optimization import OptimizationCoreset, adam

class L1KLCoreset(KLCoreset, OptimizationCoreset):

  def _max_reg_coeff(self):
    return 2*np.fabs(min(self._kl_grad_estimate().min(), 0.))
  
  def _optimize(self, w0, reg_coeff):
    def grd(w):
      g = self._kl_grad_estimate(w)
      g += reg_coeff
    return adam(self.wts, grd, opt_itrs=1000, adam_a=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
