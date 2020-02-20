import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import partial_nn_opt
from .coreset import Coreset
from ..privacy import *

def clip(x, c, axis=None):
  return (x.T/(np.linalg.norm(x, axis=axis)+1e-9)*np.clip(np.linalg.norm(x, axis=axis), 0, c)).T

class DiffPrivBatchPSVICoreset(Coreset):
  def __init__(self, data, ll_projector, opt_itrs=500, n_subsample_opt = 128, step_sched = lambda i : 1./(1.+i), 
               noise_multiplier=1.5, delta=None, init_sampler=None, gen_inits=None, l2normclip=10., **kw): 
    self.data = data
    self.ll_projector = ll_projector
    self.opt_itrs = opt_itrs
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.delta = 1./self.data.shape[0] if delta is None else delta
    self.gradclip = l2normclip 
    self.noise_mul = noise_multiplier
    self.gradcalls = self.opt_itrs
    self.init_sampler = init_sampler
    self.gen_inits = gen_inits
    self.dp = (analysis.epsilon(self.data.shape[0], self.n_subsample_opt, self.noise_mul, self.gradcalls, self.delta), self.delta)
    super().__init__(**kw)

  def _build(self, itrs, sz):
    # privately initialize points
    self._initialize(sz)
    # run gradient optimization for opt_itrs steps
    self._optimize()
  
  def _initialize(self, sz):
    # sample model parameters from pseudocoreset posterior (i.e. prior when pseudocoreset is empty)
    self.wts = self.data.shape[0]/sz*np.ones(sz)
    th0 = self.init_sampler(self.wts.shape[0], None, np.random.randint(self.data.shape[0], size=0)) # dummy init to empty set 
    self.pts = self.gen_inits(self.wts.shape[0], th0)
    self.idcs = -1*np.ones(sz)
    return
    
  def _optimize(self):
    sz = self.wts.shape[0]
    d = self.pts.shape[1]
    def grd(x):
      w = x[:sz]
      p = x[sz:].reshape((sz, d))
      vecs, sum_scaling, sub_idcs, corevecs, pgrads = self._get_projection(self.n_subsample_opt, w, p)
      #compute gradient of weights and pts
      resids = sum_scaling*self.n_subsample_opt*vecs - w.dot(corevecs) 
      wgrads = (-corevecs.dot(resids.T) / corevecs.shape[1]).transpose()
      ugrads = np.einsum('ijk,lj->lik',  -w[:,np.newaxis, np.newaxis]*pgrads, sum_scaling*self.n_subsample_opt*vecs - w.dot(corevecs))/corevecs.shape[1]
      clipped_grads = clip(np.hstack((wgrads, ugrads.reshape(ugrads.shape[0],-1))), self.gradclip)
      gauss = self.gradclip*self.noise_mul*np.random.randn(clipped_grads.shape[0], clipped_grads.shape[1])
      return np.mean(clipped_grads + gauss, axis=0)

    x0 = np.hstack((self.wts, self.pts.reshape(sz*d)))
    xf = partial_nn_opt(x0, grd, np.arange(sz), self.opt_itrs, step_sched = self.step_sched)
    self.wts = xf[:sz]
    self.pts = xf[sz:].reshape((sz, d))

  def _get_projection(self, n_subsample, w, p):
    #update the projector
    self.ll_projector.update(w, p)

    #construct a tangent space
    if n_subsample is None:
      sub_idcs = None
      vecs = self.ll_projector.project(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample

    if p.size > 0:
      corevecs, pgrads = self.ll_projector.project(p, grad=True)
    else:
      corevecs, pgrads = np.zeros((0, vecs.shape[1])), np.zeros((0, vecs.shape[1], p.shape[1]))
    return vecs, sum_scaling, sub_idcs, corevecs, pgrads
