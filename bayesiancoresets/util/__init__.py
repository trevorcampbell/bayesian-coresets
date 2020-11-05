from .opt import nn_opt
from .log import set_verbosity #, set_repeat

TOL = 1e-12
def set_tolerance(tol):
    """
    Sets the tolerance.

    Args:
        tol: (float): write your description
    """
  global TOL
  TOL = tol

