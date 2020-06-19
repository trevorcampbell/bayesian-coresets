import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset
from scipy.optimize import nnls

import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt
from .coreset import Coreset

class HOPSCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample_select=None, n_subsample_opt=None, opt_itrs=100, step_sched=lambda i : 1./(1.+i), scale_tempering_from_0_to_1 = False, **kw): 
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample_select = None if n_subsample_select is None else min(data.shape[0], n_subsample_select)
    self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.iteration_number = 1 #(if scale_tempering_from_0_to_1 is False) tracks how many iterations have been performed, to appropriately scale the tempering
    self.scale_tempering_from_0_to_1 = scale_tempering_from_0_to_1
    super().__init__(**kw)

  def _build(self, itrs):
    for i in range(itrs):
      #search for the next best point
      self._select()
      #update the weights (using a faster heuristic rather than a full optimization)
      if self.scale_tempering_from_0_to_1:
        #(assumes that _build is only called once, with "itrs" equal to the maximum allowable coreset size)
        #we want to target linearly scaling temperings of our coreset (up to 1, skipping a scaling of 0)
        tempering = np.linspace(0,1,itrs+1)[i+1]
      else:
        # scale the next hop by the current iteration count relative to the full dataset (but make sure we end at 1)
        tempering = self.iteration_number/self.data.shape[0]
        self.iteration_number += 1 
      self._hop(tempering)

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

    if self.pts.size > 0:
      corevecs = self.ll_projector.project(self.pts)
    else:
      corevecs = np.zeros((0, vecs.shape[1]))

    return vecs, sum_scaling, sub_idcs, corevecs

  def _select(self):
    vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_select, self.wts, self.pts)

    #compute the residual error
    resid = sum_scaling*vecs.sum(axis=0) - self.wts.dot(corevecs)

    #compute the correlations for the new subsample
    corrs = vecs.dot(resid) / np.sqrt((vecs**2).sum(axis=1)) / vecs.shape[1] #up to a constant; good enough for argmax
    #compute the correlations for the coreset pts (use fabs because we can decrease the weight of these)
    corecorrs = np.fabs(corevecs.dot(resid) / np.sqrt((corevecs**2).sum(axis=1))) / corevecs.shape[1] #up to a constant; good enough for argmax

    #get the best selection; if it's an old coreset pt do nothing, if it's a new point expand and initialize storage for the new pt
    if corecorrs.size == 0 or corrs.max() > corecorrs.max():
      f = sub_idcs[np.argmax(corrs)] if sub_idcs is not None else np.argmax(corrs)
      #expand and initialize storage for new coreset pt
      #need to double-check that f isn't in self.idcs, since the subsample may contain some of the coreset pts
      if f not in self.idcs:
        self.wts.resize(self.wts.shape[0]+1)
        self.idcs.resize(self.idcs.shape[0]+1)
        self.pts.resize((self.pts.shape[0]+1, self.data.shape[1]))
        self.wts[-1] = 0.
        self.idcs[-1] = f
        self.pts[-1] = self.data[f] 
    return

  # Use some quick heuristic to assign weights to the coreset (such that the next selection step, which is based on the current coreset,
  # picks a useful point)
  # Coreset weights determine the distribution from which we sample when approximating the difference between the true posterior and our current coreset, 
  # and also determine the difference between the true posterior and our current coreset (obviously). We want something reasonably accurate to the true posterior, as well as to the global optimum we could get with just these weights,
  # but also we want to calculate this quickly and with stability (we don't want our coreset weights to be too sensitive to future changes)
  def _hop(self, tempering):
    #currently, our idea is to target a tempered version of the true posterior, and solve this problem via sparse non-negative least squares
    vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, self.wts, self.pts) #alternatively, we could use the same projection as we used in select
    tempered_posterior = tempering * sum_scaling *vecs.sum(axis=0)
    self.wts = nnls(corevecs.T, tempered_posterior.T, maxiter = 100*vecs.shape[1])[0]

  #at the very end, after having selected all our coreset points, we can do a more rigorous optimization step for the weights
  def _optimize(self):
    def grd(w):
      vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts)
      resid = sum_scaling*vecs.sum(axis=0) - w.dot(corevecs)
      #output gradient of weights at idcs
      return -corevecs.dot(resid) / corevecs.shape[1]
    x0 = self.wts
    self.wts = nn_opt(x0, grd, opt_itrs = self.opt_itrs, step_sched = self.step_sched)

  def error(self):
    return 0. #TODO: implement KL estimate