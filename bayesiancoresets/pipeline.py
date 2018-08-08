from __future__ import print_function
import numpy as np

#import everything from __init__ here
from .frankwolfe import FrankWolfe
from .subsampling import ImportanceSampling, RandomSubsampling
from .fulldataset import FullDataset
from .projection import ProjectionF, Projection2
from .giga import GIGA
from .forwardstagewise import ForwardStagewise
from .pursuit import ReweightedPursuit
from .orthopursuit import OrthoPursuit
from .lar import LAR
from .pipeline import BayesianCoreset


class BayesianCoreset(object):
  def __init__(self, log_likelihoods, prior, coreset_construction_method='GIGA', discretization_method='laplace_random_projection'):
    pass

  def build(self, M):
    #discretize
    if discretization_method == 'laplace_random_projection':
      pass
    elif discretization_method == 'fixed_points':
      pass
    else:
      raise ValueError()

    #construct coreset
    if coreset_construction_method == 'GIGA':
    elif coreset_construction_method == 'FrankWolfe':
    elif coreset_construction_method == 'LAR':
    elif coreset_construction_method == 'FullDataset':
    elif coreset_construction_method == 'ReweightedPursuit':
    elif coreset_construction_method == 'OrthoPursuit':
    elif coreset_construction_method == 'ForwardStagewise':
    else:
      raise ValueError()

    #output weights
    wts = wts
    idcs = wts > 0
    alg.reset()
    return wts, idcs


