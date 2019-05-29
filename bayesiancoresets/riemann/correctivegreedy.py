import numpy as np
import warnings
from .kl import KLCoreset
from ..base.iterative import IterativeCoreset
from ..base.optimization import adam


class CorrectiveGreedyKLCoreset(KLCoreset, IterativeCoreset):

  def _search(self):
    #print('search result: ' + str(self._kl_grad(self.wts, True).argmin()))
    return self._kl_grad(self.wts, True).argmin()

  def _step(self):
    #search for FW vertex and compute line search
    f = self._search()
    #make wts[f] active
    self.wts[f] += 1e-9
    #fully optimize the active weights
    #print('before optimize: ' + str(self.wts))
    self.optimize()
    #print('after optimize: ' + str(self.wts))
    return True

