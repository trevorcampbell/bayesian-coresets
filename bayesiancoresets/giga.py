import numpy as np
from .vector import SingleGreedyVectorCoreset
from .iterative import NumericalPrecisionError

class GIGACoreset(SingleGreedyVectorCoreset):

  def _xw_unscaled(self):
    return True

  def _initialize(self):
    if self.x.shape[0] == 1:
      self.xw = self.x[0, :]
      self.wts[0] = 1.
      self.M = 1
      self.reached_numeric_limit = True
      return
    if self.x.shape[1] == 1:
      self.xw = self.xs.copy()
      self.wts[np.argmax(self.x.dot(self.xs))] = 1.0
      self.M = 1
      self.reached_numeric_limit = True
      return

  def _search(self):
    cdir = self.xs - self.xs.dot(self.xw)*self.xw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < 1e-14:
      raise NumericalPrecisionError
    cdir /= cdirnrm
    scorenums = self.x.dot(cdir) 
    scoredenoms = self.x.dot(self.xw)
    #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
    idcs = np.logical_and(scoredenoms > -1.+1e-14,  1.-scoredenoms**2 > 0.)
    #compute the norm 
    scoredenoms[idcs] = np.sqrt(1.-scoredenoms[idcs]**2)
    scoredenoms[np.logical_not(idcs)] = np.inf
    #compute the scores
    scores = scorenums/scoredenoms
    return scores.argmax()
 
  def _step_coeffs(self, f):
    gA = self.xs.dot(self.x[f,:]) - self.xs.dot(self.xw) * self.xw.dot(self.x[f,:])
    gB = self.xs.dot(self.xw) - self.xs.dot(self.x[f,:]) * self.xw.dot(self.x[f,:])
    if gA <= 0. or gB < 0:
      raise NumericalPrecisionError
    return gB/(gA+gB), gA/(gA+gB) 


