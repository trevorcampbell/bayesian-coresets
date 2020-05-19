import numpy as np
from ..util.errors import NumericalPrecisionError
from .coreset import Coreset
from ..util.timing import _tic, _toc

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

  def _build(self, sz, trace):
    for i in range(sz - self.size()):
      _tic()
      f = np.random.randint(self.data.shape[0])
      if f in self.ct_idcs:
        self.cts[self.ct_idcs.index(f)] += 1
      else:
        self.ct_idcs.append(f)
        self.cts.append(1)

      iter_t = _toc() 
      if trace:
        self._convert_cts()
        trace.append({'t': iter_t + (trace[-1]['t'] if len(trace) > 0 else 0),
		      'wts': self.wts.copy(),
                      'idcs': self.idcs.copy(),
                      'pts': self.pts.copy()
                     })

    #compute wts/idcs/pts based on cts
    self._convert_cts()

  def _convert_cts(self):
    self.wts = self.data.shape[0]*np.array(self.cts)/np.array(self.cts).sum()
    self.idcs = np.array(self.ct_idcs)
    self.pts = self.data[self.idcs]
