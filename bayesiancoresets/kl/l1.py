import numpy as np
import warnings
from .kl import KLCoreset
from ..base.optimization import OptimizationCoreset, adam

class L1KLCoreset(KLCoreset, OptimizationCoreset):

  def __init__(self, scaled = True, **kw):
    super().__init__(**kw)
    #TODO fix scales based on nat grads at w =0 
    if self.N == 0:
      self.scales = np.array([])
    elif scaled:
      self.scales = (self._sample_potentials(np.zeros(self.N))).std(axis=1)
    else:
      self.scales = np.ones(self.N)
    self.scales[self.scales == 0] = 1.

  def _max_reg_coeff(self):
    m = 2*np.fabs(min( (self._kl_grad(np.zeros(self.N))/self.scales).min(), 0.)) #extra factor of 2 to add some wiggle room
    return m if m > 0 else 1. #if the max reg coeff is 0, then all grads at w = 0 are 0; so just output 1 to avoid issues with lmbl/lmbu in base.optimizationcoreset
  
  def _optimize(self, w0, reg_coeff):
    def grd(w):
      g = self._kl_grad(w)
      g += reg_coeff*self.scales
      return g
    return adam(w0, grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)

