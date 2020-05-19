from .opt import nn_opt
from .log import set_verbosity #, set_repeat
import time

TOL = 1e-12
def set_tolerance(tol):
  global TOL
  TOL = tol


__tt = 0
def _tic():
    __tt = time.perf_counter()
def _toc():
  return time.perf_counter() - __tt
