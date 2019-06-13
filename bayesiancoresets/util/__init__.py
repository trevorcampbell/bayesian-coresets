from .opt import nn_opt
from .log import add_handler, set_verbosity, set_repeat
import logging

TOL = 1e-12
def set_tolerance(tol):
  global TOL
  TOL = tol

