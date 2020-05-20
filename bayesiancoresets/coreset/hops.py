import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset
from scipy.optimize import nnls

class HOPSCoreset(Coreset):
  def __init__(self, tangent_space_factory, opt_itrs, step_sched = lambda i : 1./(1.+i), update_single = False, **kw):
    self.tsf = tangent_space_factory
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.update_single = update_single
    self.itrs_so_far = 0
    super().__init__(**kw)

  def _build(self, itrs, sz):
    if self.size()+itrs > sz: #change this to make ure we don't run more than M itertions, if our learning rate is i/M?
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      f = self._select()
      #compute and update new weights
      self._reweight(f) 

  def _select(self):
    #construct a new tangent space for this search iteration
    self.vecs = self.tsf(self.wts, self.idcs) # made vecs a field so that reqeight can access this same projection
    #compute the correlations
    resid = self.vecs.sum(axis=0) - self.wts.dot(self.vecs[self.idcs, :])
    corrs = self.vecs.dot(resid) / np.sqrt((self.vecs**2).sum(axis=1)) / self.vecs.shape[1] #up to a constant; good enough for argmax
    #for any in the active set, just look at magnitude
    corrs[self.idcs] = np.fabs(corrs[self.idcs])
    return np.argmax(corrs)

  # (assume reweight never called before select)
  def _reweight(self, f): #had to add a  parameter to be able to scale appropriately
    if f not in self.idcs: #(this was the part of the original code that had to stay, I hope I kepteveyrthing that was needed. Wasn't sure what reweight's job was originally intended to be
      self._update(0., f)
      
    #print("shape of self.vecs: " + str(self.vecs.shape))
    scaling_vector = self.step_sched(self.itrs_so_far)*np.ones(self.vecs.shape[0]) #not sure what self.n should be
    #print("shape of scaling vector: " + str(scaling_vector.shape))
    scaling_vector[self.idcs] += ((1 - self.step_sched(self.itrs_so_far)) * self.wts)

    #put the scaled vectors in their own new tangent space
    scaled_tangent_space= scaling_vector[:, np.newaxis]*self.vecs #self.vecss is current ts?
    
    A = (scaled_tangent_space.T)[:,self.idcs]#np.matmul(self.vecs,self.vecs.T)[:,self.idcs]
    b = scaled_tangent_space.sum(axis=0)


    result = nnls(A, b, maxiter=100*self.vecs.shape[1])
    res = result[0]
    # this was when we got back N-elected 0 weights mixed in, so to adjust we may have to do
    weights = np.zeros(self.vecs.shape[0])
    weights[self.idcs] = res
    weights *= scaling_vector
    self._overwrite(weights[self.idcs], self.idcs)

    self.itrs_so_far += 1
    
  def _optimize(self): #this might be thrown off by the different step schedules we use for the rest of the problem - should there be two step schedules?
    x0 = self.wts
    def grd(w):
      #construct the tangent space
      vecs = self.tsf(w, self.idcs)
      #compute residual
      resid = vecs.sum(axis=0) - w.dot(vecs[self.idcs, :])
      #output gradient of weights at idcs
      return -vecs[self.idcs, :].dot(resid) / vecs.shape[1]
    x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
    self._update(x, self.idcs)

  def error(self):
    return 0. #TODO: implement KL estimate

