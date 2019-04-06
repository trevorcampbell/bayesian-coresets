import numpy as np
import warnings
from ..base.coreset import Coreset

class KLCoreset(Coreset): 
  def __init__(self, N, potentials, sampler, reverse=True):
    super(Coreset, self).__init__(N)
    self.potentials = potentials
    self.sampler = sampler
    self.reverse = reverse
    self.initial_scales = self._compute_scales(np.zeros(self.N))

  def _compute_scales(self, samples):
    ps = np.zeros((self.N, samples.shape[0]))
    for i in range(self.N):
      for j in range(samples.shape[0]):
        ps[i,j] = self.potentials[i](samples[j,:])   
    return ps.std(axis=1)
      
  def _forward_kl_grad_estimate(self, idcs, samples):
    pass

  def _reverse_kl_grad_estimate(self, idcs, samples):
    #first compute potentials for all involved data indices
    nzidcs = self.wts > 0
    nnz = nzidcs.sum()
    i_pots = np.zeros((samples.shape[0], len(idcs)))
    w_pots = np.zeros((samples.shape[0], nnz))
    
    for i in range(samples.shape[0]):
      for j in range(len(idcs)):
        i_pots[i, j] = potentials[idcs[j]](samples[i,:])
      for j in range(len(res_idcs)):
        r_pots[i, j] = potentials[res_idcs[j]](samples[i,:])
      for j in range(nnz):
        w_pots[i, j] = potentials[nzidcs[j]](samples[i,:])

    #now compute 



    grad = np.zeros(len(idcs))
    if self.reverse:
      for i, idx in enumerate(idcs):
        grad[i] =  
    else:
      for i, idx in enumerate(idcs):
        grad[i] =  
    return grad

  def _kl_divergence_estimate(self):
    raise NotImplementedError()

  def _log_normalization_ratio_estimate(wnum, wdenom):
    raise NotImplementedError()



