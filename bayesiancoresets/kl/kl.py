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
      self.full_wts = self.scales
      return
    if scaled:
      self.scales = self._compute_scales()
    else:
      self.scales = np.ones(self.N)
    self.scales[self.scales == 0] = 1.
    self.full_wts = np.ones(self.N)

  def weights(self):
    return self.wts/self.scales

  def error(self):
    return self._kl_estimate()

  def optimize(self):
    nzidcs = self.wts > 0
    zidcs = np.logical_not(nzidcs)
    #set inactive w gradient components to 0 
    def grd(w):
      g = self._kl_grad_estimate(w)
      g[zidcs] = 0.
      return g
    self.wts = adam(self.wts, grd, opt_itrs=1000, adam_a=1., adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8)

  def _sample_potentials(self, w, scls=None):
    if scls is None:
      scls = self.scales
    samples = self.sampler.sample(w/scls, self.n_samples)
    ps = np.zeros((self.N, samples.shape[0]))
    for i in range(self.N):
      for j in range(samples.shape[0]):
        ps[i,j] = self.potentials[i](samples[j,:])   

    ps /= scls[:, np.newaxis]

    return ps

  def _compute_scales(self):
    ps = self._sample_potentials(np.zeros(self.N), np.ones(self.N))
    return ps.std(axis=1)

  def _kl_grad_estimate(self, w, normalize=False):
    return self._reverse_kl_grad_estimate(w, normalize) if self.reverse else self._forward_kl_grad_estimate(w, normalize)

  def _kl_estimate(self):
    return self._reverse_kl_estimate() if self.reverse else self._forward_kl_estimate()
      
  def _forward_kl_grad_estimate(self, w, normalize):
    #compute two potentials
    wpots = self._sample_potentials(w)
    fpots = self._sample_potentials(self.full_wts)
    #add fpots result to the cache
    self.full_potentials_cache = (self.n_fpc*self.full_potentials_cache + fpots.shape[1]*fpots.mean(axis=1))/(self.n_fpc+fpots.shape[1])
    self.n_fpc += fpots.shape[1]
    #return grad
    return wpots.mean(axis=1) - self.full_potentials_cache

  def _reverse_kl_grad_estimate(self, w, normalize):
    pots = self._sample_potentials(w)
    residual_pots = (self.full_wts - w).dot(pots)

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
    return self._lognorm_ratio_estimate(self.wts, self.full_wts) - self._lineared_lognorm_estimate(self.wts, self.full_wts)

  def _forward_kl_estimate(self):
    return self._lognorm_ratio_estimate(self.full_wts, self.wts) - self._lineared_lognorm_estimate(self.full_wts, self.wts)

  def _linearized_lognorm_estimate(self, w0, w):
    return (w - w0).dot(self._sample_potentials(w0).mean(axis=1))

  def _lognorm_ratio_estimate(self, w0, w):
    lambdas = np.random.rand(self.n_lognorm_disc).sort()
    cusum = 0.
    for i in range(lambdas.shape[0]):
      mean_pots = self._sample_potentials((1.-lambdas[i])*w0 + lambdas[i]*w).mean(axis=1)
      cusum += ( (1.-lambdas[i])*w0 + lambdas[i]*w ).dot(mean_pots)
    return cusum / lambdas.shape[0]
    


