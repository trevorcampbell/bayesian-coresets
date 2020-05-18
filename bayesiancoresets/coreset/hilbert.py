import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class HilbertCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample=None, snnls=GIGA, **kw):

    if n_subsample is None:
      sub_idcs = None
      vecs = ll_projector.project(data)
    else:
      n_subsample = min(data.shape[0], n_subsample)
      sub_idcs = np.random.randint(data.shape[0], size=n_subsample)
      vecs = ll_projector.project(data[sub_idcs])
    vecs = vecs[np.sqrt((vecs**2).sum(axis=1))>0.,:]
    self.snnls = snnls(vecs.T, vecs.sum(axis=0))
    self.sub_idcs = sub_idcs
    self.data = data
    super().__init__(**kw)

  def reset(self):
    self.snnls.reset()
    super().reset()

  def _build(self, itrs, sz):
    if self.snnls.size()+itrs > sz:
      raise ValueError(self.alg_name + '._build(): # itrs + current size cannot exceed total desired size sz. # itr = ' + str(itrs) + ' cur sz: ' + str(self.snnls.size()) + ' desired sz: ' + str(sz))
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0] if self.sub_idcs is not None else np.where(w>0)[0]
    self.pts = self.data[self.idcs]

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0] if self.sub_idcs is not None else np.where(w>0)[0]
    self.pts = self.data[self.idcs]

  def error(self):
    return self.snnls.error()
