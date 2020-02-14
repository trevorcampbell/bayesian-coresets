import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import partial_nn_opt
from .coreset import Coreset

class BatchPSVI(Coreset):
  def __init__(self, data, ll_projector, n_subsample_opt = None, step_sched = lambda i : 1./(1.+i), **kw): 
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    super().__init__(**kw)

  def _build(self, itrs, sz):
    # initialize the points via subsampling
    init_idcs = np.random.choice(self.data.shape[0], size=sz, replace=False)
    self.pts = self.data[init_idcs]
    self.wts = self.data.shape[0]/sz*np.ones(sz)
    self.idcs = -1*np.ones(sz)

    # run gradient optimization for opt_itrs steps

  def _get_projection(self, n_subsample, w, p):
    #update the projector
    self.ll_projector.update(w, p)

    #construct a tangent space
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.choice(self.data.shape[0], size=n_subsample, replace=False)
      vecs = self.ll_projector.project(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample

    if self.pts.size > 0:
      corevecs, pgrads = self.ll_projector.project(self.pts, grads=True)
    else:
      corevecs, pgrads = np.zeros((0, vecs.shape[1])), np.zeros((0, vecs.shape[1], self.pts.shape[1]))

    return vecs, sum_scaling, sub_idcs, corevecs, pgrads

  def _optimize(self):
    sz = self.wts.shape[0]
    d = self.pts.shape[1]
    def grd(x):
      w = x[:sz]
      p = x[sz:].reshape((sz, d))
      vecs, sum_scaling, sub_idcs, corevecs, pgrads = self._get_projection(self.n_subsample_opt, w, p)

      #compute gradient of weights
      resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
      wgrad = -corevecs.dot(resid) / corevecs.shape[1]
      
      #compute gradient of pts

      #concatenate and return

      #output gradient of weights at idcs
      return g
    x0 = np.hstack((self.wts, self.pts.reshape((sz*d))))
    xf = partial_nn_opt(x0, grd, np.arange(sz), self.opt_itrs, step_sched = lambda i : 1./(i+1), b1=0.9, b2=0.99, eps=1e-8, verbose=False):
    self.wts = xf[:sz]
    self.pts = xf[sz:].reshape((sz, d))

  def error(self):
    return 0. #TODO: implement KL estimate

def grd(w):
   vecs, sum_scaling, sub_idcs, corevecs = self._get_tangent_space(self.n_subsample_opt, w, self.pts)
   

