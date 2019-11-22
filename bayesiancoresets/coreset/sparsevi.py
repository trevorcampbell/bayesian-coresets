import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset

class SparseVICoreset(Coreset):
  def __init__(self, tangent_space_factory, opt_itrs, step_sched = lambda i : 1./(1.+i), update_single = False, **kw):
    self.tsf = tangent_space_factory
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.update_single = update_single
    super().__init__(**kw)

  def _build(self, itrs, sz):
    if self.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.size()) + ' desired sz: ' + str(sz))
    for i in range(itrs):
      #search for the next best point
      f = self._select()
      #compute and update new weights
      self._reweight(f) 

  def _select(self):
    #construct a new tangent space for this search iteration
    vecs = self.tsf(self.wts, self.idcs)
    #compute the correlations
    resid = vecs.sum(axis=0) - self.wts.dot(vecs[self.idcs, :])
    corrs = vecs.dot(resid) / np.sqrt((vecs**2).sum(axis=1)) / vecs.shape[1] #up to a constant; good enough for argmax
    #for any in the active set, just look at magnitude
    corrs[self.idcs] = np.fabs(corrs[self.idcs])
    return np.argmax(corrs)

  def _reweight(self, f):
    if f not in self.idcs:
      self._update(0., f)
    fidx = np.where(self.idcs == f)[0][0]

    onef = np.zeros(self.idcs.shape[0])
    onef[fidx] = 1.
   
    if self.update_single:
      wtmp = self.wts.copy()
      wtmp[fidx] = 0.
      #since below uses nn_opt, will parametrize beta w + alpha 1_n with w[fidx] = 0
      x0 = np.array([1., self.wts[fidx]]) #scale, amt of new wt
      def grd(ab):
        #construct the tangent space
        w = ab[0]*wtmp + ab[1]*onef
        vecs = self.tsf(w, self.idcs)
        #compute residual
        resid = vecs.sum(axis=0) - w.dot(vecs[self.idcs, :])
        #output gradient of weights at idcs
        g = -vecs[self.idcs, :].dot(resid) / vecs.shape[1]
        ga = wtmp.dot(g)
        gb = g[fidx]
        return np.array([ga, gb])
      x = nn_opt(x0, grd, opt_itrs=self.opt_itrs, step_sched = self.step_sched)
      self._update(x[0]*wtmp + x[1]*onef, self.idcs)
    else:
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
    return

  def _optimize(self):
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

