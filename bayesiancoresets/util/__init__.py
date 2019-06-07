import logging
LOGLEVELS = {'error': logging.ERROR, 'warning':logging.WARNING, 'critical':logging.CRITICAL, 'info':logging.INFO, 'debug':logging.DEBUG, 'notset':logging.NOTSET}
LOGLEVEL = logging.ERROR

def verbosity(verb):
  global LOGLEVEL
  LOGLEVEL = LOGLEVELS[verb]

from .log import add_handler
from .opt import nn_opt

