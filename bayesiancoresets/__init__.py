
TOL = 1e-16
import logging
LOGFORMAT = '%(asctime)-15s %(name)s %(levelname)s --- %(message)s'
logging.basicConfig(format=LOGFORMAT)
from .tangent import FixedFiniteTangentSpace, MonteCarloFiniteTangentSpace
from .base import FullDataCoreset
from .hilbert import FrankWolfeCoreset, GIGACoreset


