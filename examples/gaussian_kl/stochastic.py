import bayesiancoresets as bc
import autograd.numpy as np
from autograd import grad
import warnings
from gaussian import *

class StochasticGaussianL1KLCoreset(bc.L1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, reverse=True, scaled=True, adam_a1=1., adam_a2=1., opt_itrs=1000):
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
                                     n_samples=n_samples, reverse=reverse, scaled=scaled, auto_above_N=False, adam_a1=adam_a1, adam_a2=adam_a2, opt_itrs=opt_itrs)

class SGL1Reverse(StochasticGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, adam_a1, adam_a2, opt_itrs, scaled=True): 
    super().__init__(x, mu0, Sig0, Sig, n_samples, True, scaled=scaled, adam_a1=adam_a1, adam_a2=adam_a2, opt_itrs=opt_itrs) 

class SGL1Forward(StochasticGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, adam_a1, adam_a2, opt_itrs, scaled=True): 
    super().__init__(x, mu0, Sig0, Sig, n_samples, False, scaled=scaled, adam_a1=adam_a1, adam_a2=adam_a2, opt_itrs=opt_itrs) 

class StochasticGaussianGreedyKLCoreset(bc.GreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, reverse=True, adam_a1=1., adam_a2=1., opt_itrs=1000):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.logdetSig = np.linalg.slogdet(self.Sig)[1]
    self.Siginv = np.linalg.inv(self.Sig)
    self.xSiginv = np.dot(self.x, self.Siginv)
    self.xSiginvx = (self.xSiginv*self.x).sum(axis=1)
    super().__init__(N = x.shape[0], potentials=lambda s : gaussian_potentials(self.Siginv, self.xSiginvx, self.xSiginv, self.logdetSig, self.x, s), 
                                     sampler=lambda w, n : gaussian_sampler(self.mu0, self.Sig0inv, self.Siginv, self.x, w, n), 
                                     n_samples=n_samples, reverse=reverse, auto_above_N=False, adam_a1=adam_a1, adam_a2=adam_a2, opt_itrs=opt_itrs)

class SGGreedyReverse(StochasticGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, adam_a1, adam_a2, opt_itrs): 
    super().__init__(x, mu0, Sig0, Sig, n_samples, True, adam_a1, adam_a2, opt_itrs) 

class SGGreedyForward(StochasticGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, n_samples, adam_a1, adam_a2, opt_itrs): 
    super().__init__(x, mu0, Sig0, Sig, n_samples, False, adam_a1, adam_a2, opt_itrs) 



