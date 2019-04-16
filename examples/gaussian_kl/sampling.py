import bayesiancoresets as bc
import autograd.numpy as np
from autograd import grad
import warnings
from gaussian import *
 

class EGS(bc.SamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=True, auto_above_N=False)

  #def _compute_scales(self):
  #  return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad(self, w, natural):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, natural):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if natural:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class SGS(bc.SamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.logdetSig = np.linalg.slogdet(self.Sig)[1]
    self.Siginv = np.linalg.inv(self.Sig)
    self.xSiginv = np.dot(self.x, self.Siginv)
    self.xSiginvx = (self.xSiginv*self.x).sum(axis=1)
    super().__init__(N = self.x.shape[0], potentials=lambda s : gaussian_potentials(self.Siginv, self.xSiginvx, self.xSiginv, self.logdetSig, self.x, s), 
                                     sampler=lambda w, n : gaussian_sampler(self.mu0, self.Sig0inv, self.Siginv, self.x, w, n), 
                                     n_samples=n_samples, reverse=True, auto_above_N=False)

class EGUS(bc.UniformSamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=True, auto_above_N=False)

  #def _compute_scales(self):
  #  return np.ones(self.x.shape[0])

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad(self, w, natural):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, natural):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if natural:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)


