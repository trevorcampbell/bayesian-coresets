import numpy as np

class BayesianTangentSpaceFactory(object):
  def __init__(self, loglike, sampler, proj_dim):
    self.proj_dim = proj_dim
    self.loglike = loglike
    self.sampler = sampler

  def __call__(self, w = None, ids = None):
    prms = self.sampler(self.proj_dim, w, ids)
    vecs = self.loglike(prms)
    vecs -= vecs.mean(axis=1)[:, np.newaxis]
    return vecs


