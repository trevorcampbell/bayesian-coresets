import numpy as np
from ..util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts = None, pts = None):
        raise NotImplementedError


class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None, w0 = None, pts0 = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(w0, pts0)

    def project(self, pts, grad=False):
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.mean(axis=1)[:,np.newaxis]
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.gradloglikelihood(pts, self.samples)
            glls -= glls.mean(axis=2)[:, :, np.newaxis]
            return lls, glls
        else:
            return lls

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)
