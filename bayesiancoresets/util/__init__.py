import logging
LOGLEVELS = {'error': logging.ERROR, 'warning':logging.WARNING, 'critical':logging.CRITICAL, 'info':logging.INFO, 'debug':logging.DEBUG, 'notset':logging.NOTSET}
LOGLEVEL = logging.ERROR
TOL = 1e-12

def set_verbosity(verb):
  global LOGLEVEL
  LOGLEVEL = LOGLEVELS[verb]

def set_tolerance(tol):
  global TOL
  TOL = tol

from .log import add_handler
from .opt import nn_opt

