import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset

class SparseVICoreset(Coreset):
    def __init__(self, data, ll_projector, n_subsample_select = None, n_subsample_opt = None, opt_itrs = 100, step_sched = lambda i : 1./(1.+i), **kw): #update_single = False, **kw):
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample_select = min(data.shape[0], n_subsample_select)
    self.n_subsample_opt = min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.update_single = update_single
    super().__init__(**kw)

  def _build(self, itrs, sz):
    if self.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      self._select()
      #update the weights
      self._optimize() 

  def _get_tangent_space(self, n_subsample, w, p):
    #update the projector
    self.ll_projector.update(w, p)

    #construct a tangent space
    if n_subsample is None:
      sub_idcs = None
      vecs = ll_projector.project(self.data)
      sum_scaling = 1.
    else:
      sub_idcs = np.random.choice(self.data.shape[0], size=n_subsample, replace=False)
      vecs = ll_projector.project(self.data[sub_idcs])
      sum_scaling = self.data.shape[0]/n_subsample
    corevecs = ll_projector.project(self.pts)

    return vecs, sum_scaling, sub_idcs, corevecs

  def _select(self):
    vecs, sum_scaling, sub_idcs, corevecs = self._get_tangent_space(self.n_subsample_select, self.wts, self.pts)

    #compute the residual error
    resid = sum_scaling*vecs.sum(axis=0) - self.wts.dot(corevecs)

    #compute the correlations for the new subsample
    corrs = vecs.dot(resid) / np.sqrt((vecs**2).sum(axis=1)) #up to a constant; good enough for argmax
    #compute the correlations for the coreset pts (use fabs because we can decrease the weight of these)
    corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) #up to a constant; good enough for argmax

    #get the best selection; if it's an old coreset pt do nothing, if it's a new point expand and initialize storage for the new pt
    if corrs.max() > corecorrs.max():
      f = sub_idcs[np.argmax(corrs)] if sub_idcs is not None else np.argmax(corrs)
      #expand and initialize storage for new coreset pt
      #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
      if f not in self.idcs:
        self.wts.resize(self.wts.shape[0]+1)
        self.idcs.resize(self.idcs.shape[0]+1)
        self.pts.resize((self.pts.shape[0]+1, self.data.shape[1]))
        self.wts[-1] = 0.
        self.idcs[-1] = sub_idcs[f] if sub_idcs is not None else f
        self.pts[-1] = self.data[sub_idcs[f]] if sub_idcs is not None else self.data[f]
    return

  #def _reweight(self, f):
  #  fidx = np.where(self.idcs == f)[0][0]
  #  onef = np.zeros(self.idcs.shape[0])
  #  onef[fidx] = 1.
  # 
  #  if self.update_single:
  #    wtmp = self.wts.copy()
  #    wtmp[fidx] = 0.
  #    #since below uses nn_opt, will parametrize beta w + alpha 1_n with w[fidx] = 0
  #    x0 = np.array([1., self.wts[fidx]]) #scale, amt of new wt
  #    def grd(ab):
  #      #construct the tangent space
  #      w = ab[0]*wtmp + ab[1]*onef
  #      vecs = self.tsf(w, self.idcs)
  #      #compute residual
  #      resid = vecs.sum(axis=0) - w.dot(vecs[self.idcs, :])
  #      #output gradient of weights at idcs
  #      g = -vecs[self.idcs, :].dot(resid) / vecs.shape[1]
  #      ga = wtmp.dot(g)
  #      gb = g[fidx]
  #      return np.array([ga, gb])
  #    x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
  #    self._update(x[0]*wtmp + x[1]*onef, self.idcs)
  #  else:
  #    x0 = self.wts
  #    def grd(w):
  #      #construct the tangent space
  #      vecs = self.tsf(w, self.idcs)
  #      #compute residual
  #      resid = vecs.sum(axis=0) - w.dot(vecs[self.idcs, :])
  #      #output gradient of weights at idcs
  #      return -vecs[self.idcs, :].dot(resid) / vecs.shape[1]
  #    x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
  #    self._update(x, self.idcs)
  #  return

  def _optimize(self):
    def grd(w):
      vecs, sum_scaling, sub_idcs, corevecs = self._get_tangent_space(self.n_subsample_opt, w, self.pts)
      resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
      #output gradient of weights at idcs
      return -vecs.dot(resid) / vecs.shape[1]
    x0 = self.wts
    self.wts = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)

  def error(self):
    return 0. #TODO: implement KL estimate
