TOL = 1e-16
import logging
LOGFORMAT = '%(asctime)-15s %(name)s %(levelname)s --- %(message)s'
logging.basicConfig(format=LOGFORMAT)
#TODO allow user to log to file
#%def set_logfile(fname):
#%  fh = logging.FileHandler(fname, mode='w')
#%  logging.basicConfig(stream=
  

from .tangent import FixedFiniteTangentSpace, MonteCarloFiniteTangentSpace
from .base import FullDataCoreset
from .hilbert import FrankWolfeCoreset, GIGACoreset


