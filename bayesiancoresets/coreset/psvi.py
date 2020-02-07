import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import opt
from .coreset import Coreset

class PseudoSparseVICoreset(Coreset):
  def __init__(self, N, tangent_space_factory, location_gradient_factory, u0, w0 = None, opt_itrs=1000, step_sched=lambda i : 1./(1.+i), **kw):
    self.tsf = tangent_space_factory
    self.lgf = location_gradient_factory
    self.step_sched = step_sched
    self.opt_itrs = opt_itrs
    self.u = u0
    self.w = w0 if w0 is not None else 
    super().__init__(**kw)

  def _build(self, itrs, sz):
    self.w, self.u = jointly_opt(w0, u0, grdw, grdu, opt_itrs=1000, step_sched = lambda i : 1./(i+1),
      b1=0.9, b2=0.99, eps=1e-8, verbose=False, idx=None):

  def _expand_tl(self, idx): # expand vectors with new initilization after selection step
    self.tl_pts[self.N+self.m] = self.pts[idx]
    self.ppts[self.m] = self.pts[idx]
    self.m += 1 # increase coreset size
    return

  def _select(self):
    ## # SELECT OPTIMAL PSEUDOPOINT INITIALIZATION
    #construct a new tangent space for this search iteration
    vecs = self.tsf(self.tl_wts[:self.N+self.m], self.tl_pts[:self.N+self.m,:], False)
    #compute the correlations
    resid = (self.alpha[:self.N+self.m]-self.tl_wts[:self.N+self.m]).dot(vecs)
    covx = vecs[:self.N].dot(resid)
    varx = np.sqrt((vecs[:self.N]**2).sum(axis=1))
    corrx = covx/varx/vecs.shape[1]
    ppt_init = np.argmax(corrx)
    # expand pseudocoreset vectors according to new ppt initilization
    self._expand_tl(ppt_init)
    return

  def _optimize(self):
    if self.update_single:
      raise NotImplementedError()
    else:
      w0 = self.pwts[:self.m]
      u0 = self.ppts[:self.m,:]
      ### # OPTIMIZE WEIGHTS
      def _grdw(w, us): # numpy computed weights gradient
        #construct the tangent space
        vecs, tvecs = self.tsf(w, us)
        return -tvecs.dot(vecs.sum(axis=0)
                         - (tvecs.T).dot(w))/ vecs.shape[1]
      def _grdps(w, us, idx): # numpy computed location gradient
        vecs, tvecs, hf = self.location_gradient_tsf(w, us, idx)
        return -w[idx]*hf.dot(vecs.sum(axis=0)
                            - (tvecs.T).dot(w))/ vecs.shape[1]
      w, u = jointly_opt(w0, u0, _grdw, _grdps, opt_itrs=self.opt_itrs,
                        step_sched=self.step_sched,idx=self.m-1)
      self._update_pwts(w)
      self._update_ppts(u)

  def _update_pwts(self, w):
    self.pwts[:self.m] = w
    self.tl_wts[self.N:self.N+self.m] = w

  def _update_ppts(self, us):
    self.tl_pts[self.N+self.m-1,:] = us[self.m-1]
    self.ppts[self.m-1,:] = us[self.m-1]

  def pseudopoints(self):
    return self.ppts

  def weights(self):
    return self.pwts, list(range(self.maxM))
