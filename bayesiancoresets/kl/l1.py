import numpy as np
import warnings
from .kl import KLCoreset
from ..base.optimization import OptimizationCoreset, OptimizationResult


#class OptimizationResult(object):
#  def __init__(self, x, f0, v0, f1, v1):
#    self.x = x
#    self.f0 = f0
#    self.v0 = v0
#    self.f1 = f1
#    self.v1 = v1

class L1KLCoreset(KLCoreset, OptimizationCoreset):
  def _max_reg_coeff(self):
    raise NotImplementedError()
  
  def _optimize(self, w0, idcs, reg_coeff):
    raise NotImplementedError()

