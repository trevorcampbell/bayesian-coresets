import numpy as np
import warnings
from ..base.coreset import Coreset
from ..base.optimization import adam
import sys

class KLCoreset(Coreset): 
  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, adam_a1 = 1., adam_a2 = 1., opt_itrs = 1000, **kw):
    super().__init__(**kw)
    self.adam_a1 = adam_a1
    self.adam_a2 = adam_a2
    self.opt_itrs = opt_itrs
    self.potentials = potentials
    self.sampler = sampler
    self.n_samples = n_samples
    self.reverse = reverse
    self.n_lognorm_disc = n_lognorm_disc
    self.n_fpc = 0
    self.full_potentials_cache = np.zeros(self.N)
    self.all_data_wts = np.ones(self.N)

  def weights(self):
    return self.wts

  def error(self):
    return self._kl()

  def optimize(self):
    nzidcs = self.wts > 0
    zidcs = np.logical_not(nzidcs)
    #set inactive w gradient components to 0 
    def grd(w):
      g = self._kl_grad(w)
      g[zidcs] = 0.
      return g
    self._update_weights(adam(self.wts, grd, opt_itrs=self.opt_itrs, adam_a1=self.adam_a1, adam_a2=self.adam_a2))

  def _sample_potentials(self, w):
    samples = self.sampler(w, self.n_samples)
    ps = self.potentials(samples)
    return ps

  def _kl(self):
    return self._reverse_kl() if self.reverse else self._forward_kl()
  
  def _kl_grad(self, w, natural=False):
    return self._reverse_kl_grad(w, natural) if self.reverse else self._forward_kl_grad(w, natural)

  def _forward_kl_grad(self, w, natural):
    #TODO implement forward nat grads
    #compute two potentials
    wpots = self._sample_potentials(w)
    fpots = self._sample_potentials(self.all_data_wts)
    #add fpots result to the cache
    self.full_potentials_cache = (self.n_fpc*self.full_potentials_cache + fpots.shape[1]*fpots.mean(axis=1))/(self.n_fpc+fpots.shape[1])
    self.n_fpc += fpots.shape[1]
    #return grad
    return wpots.mean(axis=1) - self.full_potentials_cache

  def _reverse_kl_grad(self, w, natural):
    pots = self._sample_potentials(w)
    residual_pots = (self.all_data_wts - w).dot(pots)

    #TODO fix nat grads
    num = -(pots*residual_pots).var(axis=1)
    if natural:
      denom = pots.std(axis=1) * residual_pots.std()
    else:
      denom = 1.
    if isinstance(denom, float):
      denom = 1. if denom == 0. else denom
    else:
      denom[denom == 0] = 1.

    return num / denom

  def _reverse_kl(self):
    return self._lognorm_ratio_estimate(self.wts, self.all_data_wts) - self._lineared_lognorm_estimate(self.wts, self.all_data_wts)

  def _forward_kl(self):
    return self._lognorm_ratio_estimate(self.all_data_wts, self.wts) - self._lineared_lognorm_estimate(self.all_data_wts, self.wts)

  def _linearized_lognorm_estimate(self, w0, w):
    return (w - w0).dot(self._sample_potentials(w0).mean(axis=1))

  def _lognorm_ratio_estimate(self, w0, w):
    lambdas = np.random.rand(self.n_lognorm_disc).sort()
    cusum = 0.
    for i in range(lambdas.shape[0]):
      mean_pots = self._sample_potentials((1.-lambdas[i])*w0 + lambdas[i]*w).mean(axis=1)
      cusum += ( (1.-lambdas[i])*w0 + lambdas[i]*w ).dot(mean_pots)
    return cusum / lambdas.shape[0]
    


