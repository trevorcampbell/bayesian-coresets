import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset
from ..util.timing import _tic, _toc

class HilbertCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample=None, snnls_alg=GIGA, **kw):
    super().__init__(**kw)
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample = n_subsample
    self.snnls_alg = snnls_alg
    self.sub_idcs = None 
    self.snnls = None

  def _build(self, sz, trace):
    if self.snnls is None:
      _tic()
      if self.n_subsample is None:
        #user requested to work with the whole dataset
        self.sub_idcs = np.arange(self.data.shape[0])
        vecs = self.ll_projector.project(self.data)
      else:
        #user requested to work with a subsample of the large dataset
        #randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
        self.sub_idcs = np.unique(np.random.randint(self.data.shape[0], size=self.n_subsample))
        vecs = self.ll_projector.project(self.data[self.sub_idcs])

        #remove any zero vectors; won't affect the coreset and may cause exception in snnls
        nonzero_vecs = np.sqrt((vecs**2).sum(axis=1))>0.
        self.sub_idcs = self.sub_idcs[nonzero_vecs]
        vecs = vecs[nonzero_vecs,:]

      #create the snnls object
      self.snnls = self.snnls_alg(vecs.T, vecs.sum(axis=0))
     
      #if trace is not None, the user wants detailed internal run info. append the initialization result/timing
      init_t = _toc()
      if trace is not None:
        trace.append({'t': init_t,
                      'err': self.error(),
		      'wts': self.wts.copy(),
                      'idcs': self.idcs.copy(),
                      'pts': self.pts.copy()
                     }) 

    #run the number of iterations needed to build up to size sz
    if trace is not None:
      snnls_trace = []
      self.snnls.build(sz - self.snnls.size(), trace = snnls_trace)
      self._append_snnls_trace(trace, snnls_trace)
    else:
      self.snnls.build(sz - self.snnls.size())

    #get current state
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]
    
  def _optimize(self, trace):
    if trace is not None:
      snnls_trace = []
      self.snnls.optimize(snnls_trace)
      self._append_snnls_trace(trace, snnls_trace)
    else:
      self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def error(self):
    return self.snnls.error()

  def _append_snnls_trace(self, trace, snnls_trace):
    for tr in snnls_trace:
      nnz = tr['wts'] > 0
      tr['wts'] = tr['wts'][nnz]
      tr['t'] += trace[-1]['t']
      tr['idcs'] = self.sub_idcs[nnz]
      tr['pts'] = self.data[tr['idcs']]
      #error tr['err'] needs no processing
    trace.extend(snnls_trace)
