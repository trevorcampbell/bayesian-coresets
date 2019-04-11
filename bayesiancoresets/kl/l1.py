import numpy as np
import warnings
from .kl import KLCoreset
from ..base.optimization import OptimizationCoreset, adam

class L1KLCoreset(KLCoreset, OptimizationCoreset):

  def __init__(self, N, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True):
    super().__init__(potentials=potentials, sampler=sampler, n_samples=n_samples, reverse=reverse, n_lognorm_disc=n_lognorm_disc, scaled=scaled, N=N)

  def _max_reg_coeff(self):
    m = 2*np.fabs(min(self._kl_grad(np.zeros(self.N)).min(), 0.))
    return m if m > 0 else 1. #if the max reg coeff is 0, then all grads at w = 0 are 0; so just output 1 to avoid issues with lmbl/lmbu in base.optimizationcoreset
  
  def _optimize(self, w0, reg_coeff):
    def grd(w):
      g = self._kl_grad(w)
      g += reg_coeff
      return g
    return adam(self.wts, grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
