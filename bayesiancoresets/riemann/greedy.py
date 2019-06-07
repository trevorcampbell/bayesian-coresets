import numpy as np
from ..base.optimization import adam
from ..base.iterative import GreedySingleUpdateCoreset
from ..util.errors import NumericalPrecisionError
from ..tangent.tangent import MonteCarloFiniteTangentSpace

class GreedyKLCoreset(GreedyCoreset):

  def __init__(self, N, log_likelihood, sampler, projection_dim, update_single = True):
    super().__init__(N=N) 
    self.log_likelihood = log_likelihood
    self.sampler = sampler
    self.update_single = update_single
    self.projection_dim = projection_dim
    self.T = None

  def error(self):
    #TODO KL divergence estimate
    return 0.

  def _search(self):
    #construct a new tangent space for this search iteration
    self.T = MonteCarloFiniteTangentSpace(self.log_likelihood, lambda n_samps : self.sampler(self.wts, self.idcs, n_samps), self.projection_dim)
    corrs = self.T.kl_residual_correlations(self.wts, self.idcs)
    #TODO output
    raise NotImplementedError

  def _update_weights(self, f):
    raise NotImplementedError


  def _search(self):
    return (self.T[:].dot(self.T.residual(self.wts, self.idcs)) / self.T.norms()).argmax()

  def _step_coeffs(self, f):
    nsum = self.T.norms_sum()
    nf = self.T.norms()[f]
    xw = self.T.sum_w(self.wts, self.idcs)
    xs = self.T.sum()
    xf = self.T[f]

    gammanum = (nsum/nf*xf - xw).dot(xs-xw)
    gammadenom = ((nsum/nf*xf-xw)**2).sum()

    if gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:
      raise NumericalPrecisionError('precision loss in gammanum/gammadenom: num = ' + str(gammanum) + ' denom = ' + str(gammadenom))
    return 1. - gammanum/gammadenom, nsum/nf*gammanum/gammadenom

class GreedyKLCoreset(GreedyCoreset):

  def _search(self):
    #print('search result: ' + str(self._kl_grad(self.wts, True).argmin()))
    return self._kl_grad(self.wts, True).argmin()

  def _step_coeffs(self, f):
    def grd(x):
      #wnew = alpha*wold + beta 1n
      wi = x[0]*self.wts
      wi[f] += x[1]
      g = self._kl_grad(wi)
      ga = (self.wts*g).sum()
      gb = g[f]
      return np.array([ga, gb])
    ret = adam(np.array([1., 0.]), grd, opt_itrs=self.opt_itrs, adam_a1=self.adam_a1, adam_a2=self.adam_a2)
    return ret[0], ret[1]
