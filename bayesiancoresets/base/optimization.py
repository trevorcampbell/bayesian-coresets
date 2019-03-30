import numpy as np
import warnings

class OptimizationCoreset(Coreset):
  def __init__(self):
    self.obj_cache = None #since estimating obj is expensive, keep cache for reuse

  def _build(self, M):

    #do bisection search and keep cache of results

    #template from iterative
    Mnew = self.M
    for m in range(self.M, M):
      if self.reached_numeric_limit:
        break
      stepped = self._step()
      if type(stepped) is not bool:
        raise ValueError(self.alg_name+'._build(): _step() must return a bool denoting failure or success.')
      if stepped:
        Mnew = m+1
    return Mnew

  #support stochastic: output estimate, uncertainty = variance
  def _obj(self, regularization=None):
    raise NotImplementedError()

  #support stochastic: output estimate, uncertainty = variance
  def _grad(self, idcs, regularization=None):
    raise NotImplementedError()

  def weights(self):
    return self.wts

  def error(self):
    return self._obj_estimate(True)

  def _adam(self, ):
    pass

  #TODO add prints for verbose mode
  def optimize(self, adam_a=1., opt_itrs=1000, adam_b1=0.9, adam_b2=0.99, adam_eps=1e-8, check_obj_decrease=False, verbose=False):
    #given current weight selection optimize params
    #check if adam_a is callable or numeric, opt_itrs should be int

    #optimization without regularization on fixed subset
    wi = self.wts.copy()
    nzidcs = np.logical_not(wi == 0.)
    adm_m1 = np.zeros(nzidcs.shape[0])
    adm_m2 = np.zeros(nzidcs.shape[0])
    for i in range(opt_itrs):
      g = self._grad_estimate(nzidcs)
      adm_m1 = adm_b1*adm_m1 + (1.-adm_b1)*g
      adm_m2 = adm_b2*adm_m2 + (1.-adm_b2)*g**2
      upd = adm_a(i)*adm_m1/(1.-adm_b1**(i+1))/(adm_eps + np.sqrt(adm_m2/(1.-adm_b2**(i+1))))
      wi -= upd

      #project onto w>=0
      wi = np.maximum(wi, 0.)

    #update weights to optimized version
    if check_obj_decrease:
      self.wts[nzidcs] = wi
      #TODO do 2-sample test to check whether mean decreased
    else:
      self.wts[nzidcs] = wi



