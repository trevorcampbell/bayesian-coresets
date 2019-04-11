import autograd.numpy as np
from autograd import grad
import warnings
from gaussian import *
 

class ExactGaussianL1KLCoreset(bc.L1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, reverse=True, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N = x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=reverse, scaled=scaled)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts/self.scales, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts/self.scales, reverse=True)

  def _forward_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class EGL1Reverse(ExactGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, True) 

class EGL1Forward(ExactGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig):
    super().__init__(x, mu0, Sig0, Sig, False) 

class ExactGaussianGreedyKLCoreset(bc.GreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, reverse=True, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=reverse, scaled=scaled)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts/self.scales, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts/self.scales, reverse=True)

  def _forward_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w/self.scales)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class EGGreedyReverse(ExactGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, True) 

class EGGreedyForward(ExactGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, False) 



