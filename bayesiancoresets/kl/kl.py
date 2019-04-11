import numpy as np
import warnings
from ..base.coreset import Coreset
from ..base.optimization import adam
import sys

class KLCoreset(Coreset): 
  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True, **kw):
    super().__init__(**kw)
    self.potentials = potentials
    self.sampler = sampler
    self.n_samples = n_samples
    self.reverse = reverse
    self.n_lognorm_disc = n_lognorm_disc
    self.n_fpc = 0
    self.full_potentials_cache = np.zeros(self.N)
    if self.N == 0:
      self.scales = np.array([])
    elif scaled:
      self.scales = self._compute_scales()
    else:
      self.scales = np.ones(self.N)
    self.scales[self.scales == 0] = 1.
    self.all_data_wts = self.scales

  def weights(self):
    return self.wts/self.scales

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
    self.wts = adam(self.wts, grd, opt_itrs=1000, adam_a1=1., adam_a2=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)

  def _sample_potentials(self, w, scls=None):
    if scls is None:
      scls = self.scales
    samples = self.sampler(w/scls, self.n_samples)
    ps = self.potentials(samples) / scls[:, np.newaxis]
    return ps

  def _compute_scales(self):
    if hasattr(self, '_scales_exact'):
      return self._scales_exact()
    else:
      return self._scales_estimate()

  def _kl(self):
    if (self.reverse and hasattr(self, '_reverse_kl_exact')) or (not self.reverse and hasattr(self, '_forward_kl_exact')):
      return self._reverse_kl_exact() if self.reverse else self._forward_kl_exact()
    else:
      return self._reverse_kl_estimate() if self.reverse else self._forward_kl_estimate()
  
  def _kl_grad(self, w, normalize=False):
    if (self.reverse and hasattr(self, '_reverse_kl_grad_exact')) or (not self.reverse and hasattr(self, '_forward_kl_grad_exact')):
      return self._reverse_kl_grad_exact(w, normalize) if self.reverse else self._forward_kl_grad_exact(w, normalize)
    else:
      return self._reverse_kl_grad_estimate(w, normalize) if self.reverse else self._forward_kl_grad_estimate(w, normalize)

  def _scales_estimate(self):
    ps = self._sample_potentials(np.zeros(self.N), np.ones(self.N))
    return ps.std(axis=1)

  def _forward_kl_grad_estimate(self, w, normalize):
    #compute two potentials
    wpots = self._sample_potentials(w)
    fpots = self._sample_potentials(self.all_data_wts)
    #add fpots result to the cache
    self.full_potentials_cache = (self.n_fpc*self.full_potentials_cache + fpots.shape[1]*fpots.mean(axis=1))/(self.n_fpc+fpots.shape[1])
    self.n_fpc += fpots.shape[1]
    #return grad
    return wpots.mean(axis=1) - self.full_potentials_cache

  def _reverse_kl_grad_estimate(self, w, normalize):
    pots = self._sample_potentials(w)
    residual_pots = (self.all_data_wts - w).dot(pots)

    num = -(pots*residual_pots).var(axis=1)
    if normalize:
      denom = pots.std(axis=1) * residual_pots.std()
    else:
      denom = 1.
    if isinstance(denom, float):
      denom = 1. if denom == 0. else denom
    else:
      denom[denom == 0] = 1.

    return num / denom

  def _reverse_kl_estimate(self):
    return self._lognorm_ratio_estimate(self.wts, self.all_data_wts) - self._lineared_lognorm_estimate(self.wts, self.all_data_wts)

  def _forward_kl_estimate(self):
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
    


