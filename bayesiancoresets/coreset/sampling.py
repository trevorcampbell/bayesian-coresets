import numpy as np
from ..util.errors import NumericalPrecisionError
from .coreset import Coreset

class UniformSamplingCoreset(Coreset):
  def __init__(self, data, **kw):
    super().__init__(**kw)
    self.data = data
    self.cts = []
    self.ct_idcs = [] 

  def reset(self):
    self.cts = []
    self.ct_idcs = []
    super().reset()

  def _build(self, itrs):
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
