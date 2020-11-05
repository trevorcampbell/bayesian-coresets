import numpy as np
from ..util.errors import NumericalPrecisionError
from .coreset import Coreset

class UniformSamplingCoreset(Coreset):
  def __init__(self, data, **kw):
      """
      Initialize the initialisation.

      Args:
          self: (todo): write your description
          data: (todo): write your description
          kw: (todo): write your description
      """
    super().__init__(**kw)
    self.data = data
    self.cts = []
    self.ct_idcs = [] 

  def reset(self):
      """
      Reset the internal state.

      Args:
          self: (todo): write your description
      """
    self.cts = []
    self.ct_idcs = []
    super().reset()

  def _build(self, itrs):
      """
      Builds a set of the : class : c_idcs. array.

      Args:
          self: (todo): write your description
          itrs: (todo): write your description
      """
    for i in range(itrs):
      f = np.random.randint(self.data.shape[0])
      if f in self.ct_idcs:
        self.cts[self.ct_idcs.index(f)] += 1
      else:
        self.ct_idcs.append(f)
        self.cts.append(1)
    self.wts = self.data.shape[0]*np.array(self.cts)/np.array(self.cts).sum()
    self.idcs = np.array(self.ct_idcs)
    self.pts = self.data[self.idcs]
