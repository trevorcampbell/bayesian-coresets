import bayesiancoresets as bc
import autograd.numpy as np
from autograd import grad
import warnings
from gaussian import *
 

class EGS(bc.SamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=True, scaled=scaled)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

class SGS(bc.SamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N = self.x.shape[0], potentials=lambda s : gaussian_potentials(self.Siginv, self.xSiginvx, self.xSiginv, self.logdetSig, self.x, s), 
                                     sampler=lambda w, n : gaussian_sampler(self.mu0, self.Sig0inv, self.Siginv, self.x, w, n), 
                                     n_samples=n_samples, reverse=True, scaled=scaled)

class EGUS(bc.UniformSamplingKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=True, scaled=scaled)

