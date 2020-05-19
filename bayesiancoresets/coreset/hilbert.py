import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class HilbertCoreset(Coreset):
  def __init__(self, data, ll_projector, n_subsample=None, snnls_alg=GIGA, **kw):
    super().__init__(**kw)
    self.data = data
    self.ll_projector = ll_projector
    self.n_subsample = n_subsample
    self.snnls_alg = snnls_alg
    self.sub_idcs = None 
    self.snnls = None

  def _build(self, sz, tracing):
    if self.snnls is None:
      self._tic(tracing)
      if n_subsample is None:
        #user requested to work with the whole dataset
        self.sub_idcs = np.arange(self.data.shape[0])
        vecs = ll_projector.project(self.data)
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

      #create the trace
      trace = self._toc(trace, tracing)

    #snnls algs are only expected to grow coresets; if requested sz is smaller, just log a warning and return
    if self.snnls.size() >= sz:
      self.log.warning('requested coreset of size ' + str(sz) + '; coreset is already size ' + str(self.snnls.size()) + '. Returning...')
      return

    #run the number of iterations needed to build up to size sz
    snnls_trace = self.snnls.build(sz - self.snnls.size(), tracing = tracing)

    #store the trace, converting indices where necessary
    if tracing:
      trace.extend(self._convert_trace(snnls_trace))
    
    #get current state
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

    if tracing:
      return trace
    
  def _optimize(self, tracing):
    snnls_trace = self.snnls.optimize(tracing)
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]
    return snnls_trace

  def error(self):
    return self.snnls.error()

  def _convert_trace(self, trace):
    for snap in trace:
      nnz = snap.wts > 0
      snap.wts = snap.wts[nnz]
      snap.idcs = self.sub_idcs[nnz]
      snap.pts = self.data[snap.idcs]
