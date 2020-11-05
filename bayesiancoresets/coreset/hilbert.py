import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class HilbertCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample=None, snnls=GIGA, **kw):
      """
      Initialize the dataset.

      Args:
          self: (todo): write your description
          data: (todo): write your description
          ll_projector: (todo): write your description
          n_subsample: (int): write your description
          snnls: (int): write your description
          GIGA: (todo): write your description
          kw: (todo): write your description
      """

    if n_subsample is None:
      #user requested to work with the whole dataset
      sub_idcs = np.arange(data.shape[0])
      vecs = ll_projector.project(data)
    else:
      #user requested to work with a subsample of the large dataset
      #randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
      sub_idcs = np.unique(np.random.randint(data.shape[0], size=n_subsample))
      vecs = ll_projector.project(data[sub_idcs])

      #remove any zero vectors; won't affect the coreset and may cause exception in snnls
      nonzero_vecs = np.sqrt((vecs**2).sum(axis=1))>0.
      sub_idcs = sub_idcs[nonzero_vecs]
      vecs = vecs[nonzero_vecs,:]

    self.snnls = snnls(vecs.T, vecs.sum(axis=0))
    self.sub_idcs = sub_idcs
    self.data = data
    super().__init__(**kw)

  def reset(self):
      """
      Reset the internal state.

      Args:
          self: (todo): write your description
      """
    self.snnls.reset()
    super().reset()

  def _build(self, itrs):
      """
      Build the snnls.

      Args:
          self: (todo): write your description
          itrs: (todo): write your description
      """
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def _optimize(self):
      """
      Optimize the model.

      Args:
          self: (todo): write your description
      """
    self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def error(self):
      """
      Return the error message.

      Args:
          self: (todo): write your description
      """
    return self.snnls.error()
