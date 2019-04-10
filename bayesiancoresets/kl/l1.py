import numpy as np
import warnings
from .kl import KLCoreset
from ..base.optimization import OptimizationCoreset, adam

class L1KLCoreset(KLCoreset, OptimizationCoreset):

  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True):
    smp = sampler.sample(1)
    ps = potentials(np.atleast_2d(smp))
    if ps.ndim != 2:
      raise ValueError(self.alg_name + ".__init__(): potentials must return an N x (num samples) 2d numpy array")
    super().__init__(potentials=potentials, sampler=sampler, n_samples=n_samples, reverse=reverse, n_lognorm_disc=n_lognorm_disc, scaled=scaled, N=ps.shape[0])

  def _max_reg_coeff(self):
    return 2*np.fabs(min(self._kl_grad_estimate(np.zeros(self.N)).min(), 0.))
  
  def _optimize(self, w0, reg_coeff):
    def grd(w):
      g = self._kl_grad_estimate(w)
      g += reg_coeff
    return adam(self.wts, grd, opt_itrs=10000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)
