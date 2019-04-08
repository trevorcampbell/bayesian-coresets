import numpy as np
import warnings
from ..base.coreset import Coreset
from ..base.optimization import adam

class KLCoreset(Coreset): 
  def __init__(self, potentials, sampler, n_samples, reverse=True, n_lognorm_disc = 100, scaled=True, normalized = True, **kw):
    super().__init__(**kw)
    self.potentials = potentials
    self.sampler = sampler
    self.n_samples = n_samples
    self.reverse = reverse
    if scaled:
      self.scales = self._compute_scales(S)
      self.full_wts = self.scales
    else:
      self.scales = np.ones(self.N)
      self.full_wts = np.ones(self.N)
    self.normalized = normalized
    self.full_potentials_cache = np.zeros(self.N)
    self.n_fpc = 0
    self.n_lognorm_disc = n_lognorm_disc

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
    if not scls:
      scls = self.scales
    samples = self.sampler.sample(w/scls, self.n_samples)
    ps = np.zeros((self.N, samples.shape[0]))
    for i in range(self.N):
      for j in range(samples.shape[0]):
        ps[i,j] = self.potentials[i](samples[j,:])   
    ps /= scls

    return ps

  def _compute_scales(self):
    ps = self._sample_potentials(np.zeros(self.N), scls = np.ones(self.N))
    return ps.std(axis=1)

  def _kl_grad_estimate(self, w):
    return self._reverse_kl_grad_estimate(w) if self.reverse else self._forward_kl_grad_estimate(w)

  def _kl_estimate(self):
    return self._reverse_kl_estimate() if self.reverse else self._forward_kl_estimate()
      
  def _forward_kl_grad_estimate(self, w):
    #compute two potentials
    wpots = self._sample_potentials(w)
    fpots = self._sample_potentials(self.full_wts)
    #add fpots result to the cache
    self.full_potentials_cache = (self.n_fpc*self.full_potentials_cache + fpots.shape[1]*fpots.mean(axis=1))/(self.n_fpc+fpots.shape[1])
    self.n_fpc += fpots.shape[1]
    #return grad
    return wpots.mean(axis=1) - self.full_potentials_cache

  def _reverse_kl_grad_estimate(self, w):
    pots = self._sample_potentials(w)
    residual_pots = (self.full_wts - w).dot(pots)

    num = -(pots*residual_pots).var(axis=1)
    if self.normalized:
      denom = pots.std(axis=1) * residual_pots.std()
    else:
      denom = 1.

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
    


