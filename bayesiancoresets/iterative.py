import numpy as np
import warnings
from .coreset import Coreset

class IterativeCoreset(Coreset):
  def _build(self, M):
    Mnew = self.M
    for m in range(self.M, M):
      if self.reached_numeric_limit:
        break
      stepped = self._step()
      if stepped:
        Mnew = m+1
    return Mnew

  def _step(self):
    raise NotImplementedError()

class StepFailureError(Exception):
  pass

class SearchFailureError(Exception):
  pass

class GreedySingleUpdate(IterativeCoreset):

  def _step(self):
    #search for the next best point and step length
    try:
      f = self._search()
    except SearchFailureError:
      print('Greedy next point selection failed')
    alpha, beta = self._step_coeffs(f) 

    #if the line search is invalid, possibly reached numeric limit
    if alpha is None:
      #try recomputing xw from scratch and rerunning search
      self.xw = self.wts.dot(self.x)
      if self._xw_unscaled():
        self._renormalize()
      f = self._search()
      alpha, beta = self._step_coeffs(f) 

      #if it's still no good, we've reached the numeric limit
      if alpha is None: #gammanum < 0. or gammadenom == 0. or gammanum > gammadenom:  
        self.reached_numeric_limit = True
        return False

    #update the weights
    self.wts *= alpha
    #it's possible wts[f] becomes negative if beta approx -wts[f], so threshold
    self.wts[f] = max(self.wts[f]+beta, 0)
    #apply the same update to xw
    if use_cached_xw:
      self.xw = alpha*self.xw + beta*self.x[f, :]
    else:
      self.xw = self.wts.dot(self.x)

    if self._xw_unscaled():
      self._renormalize()

    return True

  def _search(self):
    raise NotImplementedError()

  def _step_coeffs(self, f):
    raise NotImplementedError()

  def _step_prepare_retry(self):
    raise NotImplementedError()
  
  def _search_prepare_retry(self):
    raise NotImplementedError()

  



