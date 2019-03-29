import numpy as np
import warnings

class KLCoreset(Coreset): 
  def __init__(self, N, potentials, sampler, reverse=True):
    super(DivergenceCoreset, self).__init__(N)

    potentials(sample)

    sampler.sample(wts)

  def _kl_divergence_estimate(self):
    pass
  
  def _kl_grad_estimate(self):
    pass

  def _log_normalization_ratio_estimate(wnum, wdenom):
    #TODO thermodynamic int
    pass



